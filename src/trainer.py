import os
import umap
import wandb
import torch
import numpy as np
import pandas as pd 
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, average_precision_score


def save_confusion_matrix_png(cm, output_path, title):
    """Save the confusion matrix as a PNG image."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        title=title,
        xlabel='Predicted label',
        ylabel='True label',
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0])
    )

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha='center',
                va='center',
                color='white' if cm[i, j] > threshold else 'black'
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_confusion_matrix_png_gender(cm, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title, xlabel='Predicted label', ylabel='True label',
           xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def is_improvement(current_loss, best_loss, min_delta):
    return current_loss < (best_loss - min_delta)

class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=5e-5,
        num_classes=6,
        save_name="best_spatial_model.pth",
        test_loader=None,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        save_path="runs/spatial_experiment",
        project_name="K-Print",
        epochs=100,
        subfix=[],
        batch_size=8
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_name = save_name
        self.save_path = save_path
        self.project_name = project_name
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.epochs = epochs
        self.subfix = subfix
        self.batch_size = batch_size
        
        self.save_path = os.path.join(save_path, "scanner", self.subfix[0] if self.subfix else "")
        os.makedirs(self.save_path, exist_ok=True)
        
        os.makedirs(self.save_path, exist_ok=True)
        wandb.init(
            project=self.project_name,
            name=os.path.splitext(self.save_name)[0].replace("best_", ""),
            group="Scanner Classification Spatial Space",
            dir=self.save_path,
            config={
                "learning_rate": learning_rate,
                "epochs_patience": early_stopping_patience,
                "num_classes": num_classes,
                "domain": "spatial",
                "batch_size": self.batch_size
            },
            tags=["spatial", "classification", "resnet"] + self.subfix
        )
        
    def _calculate_metrics(self, targets, preds, probs):
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)  
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        spec_list = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = np.mean(spec_list)
            
        try:
            roc_auc = roc_auc_score(targets, probs[:, 1] if self.num_classes == 2 else probs, 
                                    multi_class='ovr' if self.num_classes > 2 else 'raise')
        except ValueError:
            roc_auc = float('nan')
            
        return acc, f1, precision, recall, specificity, recall, roc_auc, cm

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)  
                preds = torch.argmax(probs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
        loss = running_loss / len(self.train_loader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
            
        return {"Loss": loss, "Accuracy": acc, "F1": f1, "Precision": prec, 
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, "ROC_AUC": auc}

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        loss_val = running_loss / len(dataloader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
        
        return {"Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec,
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, 
                "ROC_AUC": auc, "Confusion_Matrix": cm}

    def evaluate_best_on_test(self):
        if self.test_loader is None: return None
        
        test_cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix_test.png")
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, self.save_name), map_location=self.device))
        
        test_metrics = self.evaluate(self.test_loader)
        
        print("\n" + "="*50 + "\n TEST SET RESULTS (Best Model)\n" + "="*50)
        print(f"Loss: {test_metrics['Loss']:.4f} | Acc: {test_metrics['Accuracy']:.4f} | "
              f"F1: {test_metrics['F1']:.4f} | Precision: {test_metrics['Precision']:.4f} | "
              f"Recall: {test_metrics['Recall']:.4f} | ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
              f"Spec: {test_metrics['Specificity']:.4f} | Sens: {test_metrics['Sensitivity']:.4f}")
        
        final_cm = test_metrics['Confusion_Matrix']
        print("\nConfusion Matrix (Test Set):")
        print(final_cm)
        save_confusion_matrix_png(final_cm, test_cm_png_path, "Confusion Matrix - Best Model Test Set")
        
        wandb.run.summary["Test/loss"] = test_metrics['Loss']
        wandb.run.summary["Test/accuracy"] = test_metrics['Accuracy']
        wandb.run.summary["Test/f1"] = test_metrics['F1']
        wandb.run.summary["Test/precision"] = test_metrics['Precision']
        wandb.run.summary["Test/recall"] = test_metrics['Recall']
        wandb.run.summary["Test/specificity"] = test_metrics['Specificity']
        wandb.run.summary["Test/sensitivity"] = test_metrics['Sensitivity']
        wandb.run.summary["Test/roc_auc"] = test_metrics['ROC_AUC']
        wandb.run.summary["Test/Confusion_Matrix"] = wandb.Image(test_cm_png_path)
        
        return test_metrics

    def fit(self):
        final_cm = None
        cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix.png")
        csv_history_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_training_history.csv")
        metrics_history = []
        
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if not is_improvement(val_metrics['Loss'], self.best_val_loss, self.early_stopping_min_delta):
                wandb.log({
                    "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                    "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                    "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                    "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                    "Precision/Train": train_metrics['Precision'], "Precision/Validation": val_metrics['Precision'],
                    "Recall/Train": train_metrics['Recall'], "Recall/Validation": val_metrics['Recall'],
                    "Sensitivity/Train": train_metrics['Sensitivity'], "Sensitivity/Validation": val_metrics['Sensitivity'],
                    "Specificity/Train": train_metrics['Specificity'], "Specificity/Validation": val_metrics['Specificity'],
                    "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC']
                })
                self.epochs_without_improvement += 1
            else:
                save_message = "  [Save: New Best Model! | CM PNG updated]"
                self.best_val_loss = val_metrics['Loss']
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_path, self.save_name))
                
                final_cm = val_metrics['Confusion_Matrix']
                save_confusion_matrix_png(final_cm, cm_png_path, "Confusion Matrix - Best Model")
                
                wandb.log({
                    "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                    "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                    "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                    "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                    "Precision/Train": train_metrics['Precision'], "Precision/Validation": val_metrics['Precision'],
                    "Recall/Train": train_metrics['Recall'], "Recall/Validation": val_metrics['Recall'],
                    "Sensitivity/Train": train_metrics['Sensitivity'], "Sensitivity/Validation": val_metrics['Sensitivity'],
                    "Specificity/Train": train_metrics['Specificity'], "Specificity/Validation": val_metrics['Specificity'],
                    "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC'],
                    "Confusion Matrix (Best)": wandb.Image(cm_png_path)
                })
                
                metrics_history.append({
                    "Epoch": epoch + 1, "LR": current_lr,
                    "Train_Loss": train_metrics['Loss'], "Val_Loss": val_metrics['Loss'],
                    "Train_Acc": train_metrics['Accuracy'], "Val_Acc": val_metrics['Accuracy'],
                    "Train_F1": train_metrics['F1'], "Val_F1": val_metrics['F1'],
                    "Train_AUC": train_metrics['ROC_AUC'], "Val_AUC": val_metrics['ROC_AUC']
                })
                pd.DataFrame(metrics_history).to_csv(csv_history_path, index=False)
            
            print(f"Epoch [{epoch+1}/{self.epochs}] - LR: {current_lr:.6f} | ")
            print(f"Train Loss: {train_metrics['Loss']:.4f} | Val Loss: {val_metrics['Loss']:.4f} | "
                  f"Train Acc: {train_metrics['Accuracy']:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | "
                  f"Train F1: {train_metrics['F1']:.4f} | Val F1: {val_metrics['F1']:.4f}{save_message if 'save_message' in locals() else ''}")
            print(f"          -> Train AUC: {train_metrics['ROC_AUC']:.4f} | Val AUC: {val_metrics['ROC_AUC']:.4f} | "
                  f"Train Spec: {train_metrics['Specificity']:.4f} | Val Spec: {val_metrics['Specificity']:.4f}")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping activated at epoch {epoch+1}")
                break
        
        print("\n" + "="*40 + "\n CONFUSION MATRIX (BEST MODEL)\n" + "="*40)
        print(final_cm)
        self.evaluate_best_on_test()
        wandb.finish()

class KSpaceModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=5e-5,
        num_classes=6,
        save_name="best_kspace_model.pth",
        test_loader=None,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        save_path="runs/kspace_experiment",
        project_name="K-Print",
        epochs=100,
        subfix=[],
        batch_size=8,
        label_smoothing=None,
        model_kwargs=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_name = save_name
        self.project_name = project_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing is not None else nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.05)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.subfix = subfix
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs or {}   
        
        self.best_val_loss = float('inf') 
        self.best_val_metric = 0.0     
        
        self.use_metric_learning_saving = "ArcFace" in self.model.__class__.__name__
        
        self.save_path = os.path.join(save_path, "scanner", self.subfix[0] if self.subfix else "", self.subfix[-1] if self.subfix else "")
        os.makedirs(self.save_path, exist_ok=True)
        
        config={
            "learning_rate": learning_rate,
            "epochs_patience": early_stopping_patience,
            "num_classes": num_classes,
            "domain": "k-space",
            "batch_size": self.batch_size
        }

        if self.model_kwargs:
            config.update(self.model_kwargs)
        
        wandb.init(
            project=self.project_name,
            name=os.path.splitext(self.save_name)[0].replace("best_", ""),
            group="Scanner Classification K-Space",
            dir=self.save_path,
            config=config,
            tags=["k-space", "classification"] + self.subfix
        )

    def _calculate_metrics(self, targets, preds, probs):
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        spec_list = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = np.mean(spec_list)
            
        try:
            roc_auc = roc_auc_score(targets, probs[:, 1] if self.num_classes == 2 else probs, 
                                    multi_class='ovr' if self.num_classes > 2 else 'raise')
        except ValueError:
            roc_auc = float('nan')
            
        return acc, f1, precision, recall, specificity, recall, roc_auc, cm

    def plot_latent_space(self, dataloader, filename="latent_space_clusters.png"):
        print("\n=> Generating Latent Space Visualization (t-SNE & UMAP)...")
        
        if hasattr(self.model, 'head'):
            original_layer = self.model.head
            self.model.head = nn.Identity()
            layer_name = 'head'
        elif hasattr(self.model, 'fc'):
            original_layer = self.model.fc
            self.model.fc = nn.Identity()
            layer_name = 'fc'
        else:
            print("[Error] Unable to find the final layer ('head' or 'fc') to extract features.")
            return

        self.model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                features = self.model(inputs)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print("   - Calculating t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(all_features)

        print("   - Calculating UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_results = reducer.fit_transform(all_features)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        palette = sns.color_palette("tab10", len(np.unique(all_labels)))

        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=all_labels, palette=palette, legend=False,
            alpha=0.8, s=80, ax=axes[0]
        )
        axes[0].set_title("t-SNE (Local Structure)", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(
            x=umap_results[:, 0], y=umap_results[:, 1],
            hue=all_labels, palette=palette, legend="full",
            alpha=0.8, s=80, ax=axes[1]
        )
        axes[1].set_title("UMAP (Global Topology)", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        axes[1].legend(title="Device Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle("Latent Space of the MLP-Mixer", fontsize=16, y=1.02, fontweight='bold')
        plt.tight_layout()

        save_file_path = os.path.join(self.save_path, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        print(f"   - Plot saved in: {save_file_path}")

        if wandb.run is not None:
            wandb.log({"Test_Set/Latent_Space_Clusters": wandb.Image(save_file_path)})

        plt.close()

        setattr(self.model, layer_name, original_layer)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        for batch in self.train_loader:
            inputs = batch[0].to(self.device) 
            targets = batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            if "ArcFace" in self.model.__class__.__name__:
                    outputs = self.model(inputs, targets)
            else:
                outputs = self.model(inputs)
    
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)  
                preds = torch.argmax(probs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
        loss = running_loss / len(self.train_loader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
            
        return {"Loss": loss, "Accuracy": acc, "F1": f1, "Precision": prec, 
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, "ROC_AUC": auc}

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device) 
                targets = batch[1].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        loss_val = running_loss / len(dataloader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
        
        return {"Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec,
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, 
                "ROC_AUC": auc, "Confusion_Matrix": cm}

    def evaluate_best_on_test(self):
        if self.test_loader is None: return None
        
        test_cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix_test.png")
        latent_plot_name = f"{os.path.splitext(self.save_name)[0]}_latent_space_test.png" 
        
        checkpoint = torch.load(os.path.join(self.save_path, self.save_name), map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        try:
            self.plot_latent_space(self.test_loader, filename=latent_plot_name)
        except Exception as e:
            print(f"\n[Warning] Impossible to generate the latent space plot: {e}")

        test_metrics = self.evaluate(self.test_loader)
        
        print("\n" + "="*50 + "\n TEST SET RESULTS (Best K-Space Model)\n" + "="*50)
        print(f"Loss: {test_metrics['Loss']:.4f} | Acc: {test_metrics['Accuracy']:.4f} | "
              f"F1: {test_metrics['F1']:.4f} | Precision: {test_metrics['Precision']:.4f} | "
              f"Recall: {test_metrics['Recall']:.4f} | ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
              f"Spec: {test_metrics['Specificity']:.4f} | Sens: {test_metrics['Sensitivity']:.4f}")
        
        final_cm = test_metrics['Confusion_Matrix']
        save_confusion_matrix_png(final_cm, test_cm_png_path, "Confusion Matrix - Best Test Model")
        
        if wandb.run is not None:
            wandb.run.summary["Test/Loss"] = test_metrics['Loss']
            wandb.run.summary["Test/Accuracy"] = test_metrics['Accuracy']
            wandb.run.summary["Test/f1"] = test_metrics['F1']
            wandb.run.summary["Test/precision"] = test_metrics['Precision']
            wandb.run.summary["Test/recall"] = test_metrics['Recall']
            wandb.run.summary["Test/specificity"] = test_metrics['Specificity']
            wandb.run.summary["Test/sensitivity"] = test_metrics['Sensitivity']
            wandb.run.summary["Test/ROC_AUC"] = test_metrics['ROC_AUC']
            wandb.run.summary["Test/Confusion_Matrix"] = wandb.Image(test_cm_png_path)
            wandb.log({"Test/Confusion_Matrix": wandb.Image(test_cm_png_path)})
        
        return test_metrics

    def fit(self, epochs=None):
        if epochs is not None: self.epochs = epochs
        final_cm = None
        cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix.png")
        csv_history_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_training_history.csv")
        metrics_history = []
        
        for epoch in range(self.epochs):
            
            save_message = ""
            
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics_history.append({
                "Epoch": epoch + 1, 
                "LR": current_lr,
                "Train_Loss": train_metrics['Loss'], "Val_Loss": val_metrics['Loss'],
                "Train_Acc": train_metrics['Accuracy'], "Val_Acc": val_metrics['Accuracy'],
                "Train_F1": train_metrics['F1'], "Val_F1": val_metrics['F1'],
                "Train_AUC": train_metrics['ROC_AUC'], "Val_AUC": val_metrics['ROC_AUC']
            })
            pd.DataFrame(metrics_history).to_csv(csv_history_path, index=False)
            
            is_new_best = False
            
            if self.use_metric_learning_saving:
                current_score = val_metrics['Accuracy'] 
                if current_score > self.best_val_metric + self.early_stopping_min_delta:
                    self.best_val_metric = current_score
                    is_new_best = True
            else:
                current_loss = val_metrics['Loss']

                if current_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = current_loss
                    is_new_best = True

            if not is_new_best:
                wandb.log({
                    "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                    "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                    "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                    "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                    "Precision/Train": train_metrics['Precision'], "Precision/Validation": val_metrics['Precision'],
                    "Recall/Train": train_metrics['Recall'], "Recall/Validation": val_metrics['Recall'],
                    "Sensitivity/Train": train_metrics['Sensitivity'], "Sensitivity/Validation": val_metrics['Sensitivity'],
                    "Specificity/Train": train_metrics['Specificity'], "Specificity/Validation": val_metrics['Specificity'],
                    "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC']
                })
                self.epochs_without_improvement += 1
            else:
                save_message = "  [Saving: New Best Model! | CM PNG Updated]"
                self.epochs_without_improvement = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'best_val_metric': self.best_val_metric,
                    'model_kwargs': self.model_kwargs
                }
                torch.save(checkpoint, os.path.join(self.save_path, self.save_name))
                
                final_cm = val_metrics['Confusion_Matrix']
                save_confusion_matrix_png(final_cm, cm_png_path, "Confusion Matrix - Best Model")
                
                wandb.log({
                    "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                    "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                    "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                    "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                    "Precision/Train": train_metrics['Precision'], "Precision/Validation": val_metrics['Precision'],
                    "Recall/Train": train_metrics['Recall'], "Recall/Validation": val_metrics['Recall'],
                    "Sensitivity/Train": train_metrics['Sensitivity'], "Sensitivity/Validation": val_metrics['Sensitivity'],
                    "Specificity/Train": train_metrics['Specificity'], "Specificity/Validation": val_metrics['Specificity'],
                    "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC'],
                })
                
                
            print(f"Epoch [{epoch+1}/{self.epochs}] - LR: {current_lr:.6f} | "
                  f"Train Loss: {train_metrics['Loss']:.4f} | Val Loss: {val_metrics['Loss']:.4f} | "
                  f"Train Acc: {train_metrics['Accuracy']:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | "
                  f"Train F1: {train_metrics['F1']:.4f} | Val F1: {val_metrics['F1']:.4f}{save_message if 'save_message' in locals() else ''}")
            print(f"          -> Train AUC: {train_metrics['ROC_AUC']:.4f} | Val AUC: {val_metrics['ROC_AUC']:.4f} | "
                  f"Train Spec: {train_metrics['Specificity']:.4f} | Val Spec: {val_metrics['Specificity']:.4f}")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping activated at epoch {epoch+1}")
                break
        
        self.evaluate_best_on_test()
        wandb.finish()
        
class FangTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        num_classes=6,
        save_path="best_fang_model.pth",
        test_loader=None,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            self.optimizer.step()

            if hasattr(self.model, 'constrained_layer'):
                self.model.constrained_layer.apply_clipping_correction()
                
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(self.train_loader.dataset)

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        loss_val = running_loss / len(dataloader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds, labels=range(self.num_classes))
        
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = float('nan')
            
        return {
            "Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": precision,
            "Recall": recall, "ROC_AUC": roc_auc, "Confusion_Matrix": cm
        }

    def evaluate_best_on_test(self):
        if self.test_loader is None:
            print("Test loader not provided: skipping final evaluation on test set.")
            return None

        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))

        test_metrics = self.evaluate(self.test_loader)
        print("\n" + "="*40)
        print(" TEST SET RESULTS (Best MISLnet Model)")
        print("="*40)
        print(
            f"Loss: {test_metrics['Loss']:.4f} | "
            f"Acc: {test_metrics['Accuracy']:.4f} | "
            f"F1: {test_metrics['F1']:.4f} | "
            f"Precision: {test_metrics['Precision']:.4f} | "
            f"Recall: {test_metrics['Recall']:.4f} | "
            f"ROC-AUC: {test_metrics['ROC_AUC']:.4f}"
        )
        print(test_metrics["Confusion_Matrix"])
        return test_metrics

    def fit(self, epochs):
        final_cm = None
        cm_png_path = os.path.splitext(self.save_path)[0] + "_confusion_matrix.png"
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            self.scheduler.step(val_metrics['Loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            current_val_loss = val_metrics['Loss']
            
            save_message = ""
            if is_improvement(current_val_loss, self.best_val_loss, self.early_stopping_min_delta):
                self.best_val_loss = current_val_loss
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.save_path)
                save_message = f"  [Saving: New Best Model! | CM PNG: {cm_png_path}]"
                final_cm = val_metrics['Confusion_Matrix']
                save_confusion_matrix_png(final_cm, cm_png_path, "Confusion Matrix - Best MISLnet Model")
            else:
                self.epochs_without_improvement += 1
            
            print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {current_val_loss:.4f} | "
                  f"Acc: {val_metrics['Accuracy']:.4f} | F1: {val_metrics['F1']:.4f} | "
                  f"ROC-AUC: {val_metrics['ROC_AUC']:.4f}{save_message}")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"Early stopping activated at epoch {epoch+1}: "
                    f"no improvement in validation loss for {self.early_stopping_patience} epochs."
                )
                break
        
        print("\n" + "="*40)
        print(" CONFUSION MATRIX (Best MISLnet Model)")
        print("="*40)
        print(final_cm)
        print(f"Confusion matrix PNG saved in: {cm_png_path}")
        self.evaluate_best_on_test()

class GenderModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=5e-5,
        num_classes=2, 
        save_name="best_gender_model.pth",
        test_loader=None,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        save_path="runs/gender_experiment",
        project_name="K-Print Gender Classification",
        epochs=100,
        subfix=[],
        batch_size=8,
        use_wandb=True,
        model_kwargs=None
    ):
        self.use_wandb = use_wandb
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_name = save_name
        self.project_name = project_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0
        self.epochs = epochs
        self.subfix = subfix
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs or {}

        self.save_path = os.path.join(save_path, "gender_classification", self.subfix[0] if self.subfix else "", self.subfix[1] if len(self.subfix) > 1 else "")
        os.makedirs(self.save_path, exist_ok=True)
        
        self.best_val_loss = float('inf')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        config = {
            "learning_rate": self.lr,
            "epochs_patience": early_stopping_patience,
            "num_classes": self.num_classes,
            "domain": "gender_kspace",
            "batch_size": self.batch_size,
            "loss_function": "CrossEntropy"
        }
        if self.model_kwargs:
            config.update(self.model_kwargs)
            
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=os.path.splitext(self.save_name)[0].replace("best_", ""),
                group="Model Gender Classification MSDA",
                dir=self.save_path,
                config=config,
                tags=["gender", "classification", "MSDA"] + self.subfix
            )

    def _calculate_metrics(self, targets, preds, probs):
        """Calculate the metrics. Since it's a balanced binary task (M/F), macro average is appropriate."""
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0) 
        
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        if self.num_classes == 2 and cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            specificity, sensitivity = float('nan'), recall
            
        try:
            roc_auc = roc_auc_score(targets, probs[:, 1] if self.num_classes == 2 else probs, 
                                    multi_class='ovr' if self.num_classes > 2 else 'raise')
        except ValueError:
            roc_auc = float('nan')
            
        return acc, f1, precision, recall, specificity, sensitivity, roc_auc, cm

    def _save_confusion_matrix_png(self, cm, save_path, title="Confusion Matrix"):
        """Generate and save the confusion matrix image."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []
        
        for batch in self.train_loader:
            inputs = batch[0].to(self.device)
       
            targets = batch[1].to(self.device).long() 
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)  
                preds = torch.argmax(probs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
        loss_val = running_loss / len(self.train_loader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
            
        return {"Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec, 
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, "ROC_AUC": auc}

    def evaluate(self, dataloader, disaggregate=False):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs, all_scanners = [], [], [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device).long() 
                
                if disaggregate:
                    scanners = batch[2] 
                    all_scanners.extend(scanners.cpu().numpy() if torch.is_tensor(scanners) else scanners)
                    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        loss_val = running_loss / len(dataloader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
        
        metrics = {
            "Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec,
            "Recall": rec, "Specificity": spec, "Sensitivity": sens, 
            "ROC_AUC": auc, "Confusion_Matrix": cm
        }
        
        if disaggregate and all_scanners:
            metrics["Scanner_Breakdown"] = {}
            unique_scanners = np.unique(all_scanners)
            for scanner in unique_scanners:
                idx = np.array(all_scanners) == scanner
                s_targets = np.array(all_targets)[idx]
                s_preds = np.array(all_preds)[idx]
                s_probs = np.array(all_probs)[idx]
                
                if len(s_targets) > 0:
                    s_acc, s_f1, _, _, _, _, s_auc, _ = self._calculate_metrics(s_targets, s_preds, s_probs)
                    metrics["Scanner_Breakdown"][scanner] = {
                        "Accuracy": s_acc, "F1": s_f1, "ROC_AUC": s_auc, "Count": len(s_targets)
                    }
                    
        return metrics

    def evaluate_best_on_test(self):
        if self.test_loader is None: return None
        
        test_cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix_test.png")

        checkpoint = torch.load(os.path.join(self.save_path, self.save_name), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(self.test_loader, disaggregate=True)
        
        print("\n" + "="*50)
        print(" TEST SET RESULTS (Best MSDA Model)")
        print("="*50)
        print(f"Global Loss: {test_metrics['Loss']:.4f} | Acc: {test_metrics['Accuracy']:.4f} | "
              f"F1: {test_metrics['F1']:.4f} | Precision: {test_metrics['Precision']:.4f} | "
              f"Recall: {test_metrics['Recall']:.4f} | ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
              f"Spec: {test_metrics['Specificity']:.4f} | Sens: {test_metrics['Sensitivity']:.4f}")
        
        final_cm = test_metrics['Confusion_Matrix']
        self._save_confusion_matrix_png(final_cm, test_cm_png_path, "Confusion Matrix - Test Set (MSDA)")
        
        if self.use_wandb and wandb.run is not None:
            wandb.run.summary["Test/loss"] = test_metrics['Loss']
            wandb.run.summary["Test/accuracy"] = test_metrics['Accuracy']
            wandb.run.summary["Test/f1"] = test_metrics['F1']
            wandb.run.summary["Test/precision"] = test_metrics['Precision']
            wandb.run.summary["Test/recall"] = test_metrics['Recall']
            wandb.run.summary["Test/specificity"] = test_metrics['Specificity']
            wandb.run.summary["Test/sensitivity"] = test_metrics['Sensitivity']
            wandb.run.summary["Test/roc_auc"] = test_metrics['ROC_AUC']
            wandb.run.summary["Test/Confusion_Matrix"] = wandb.Image(test_cm_png_path)
        
        if "Scanner_Breakdown" in test_metrics:
            label_mapping = {
                0: 'Intera', 1: 'SIGNA EXCITE', 2: 'SIGNA HDx', 3: 'Verio', 
                4: 'Achieva', 5: 'TrioTim'
            }
            print("\n--- Detailed Analysis per Scanner (Target Domain) ---")
            breakdown_list = []
            for scanner, m in test_metrics["Scanner_Breakdown"].items():
                mfr_name = label_mapping.get(scanner, f"Unknown_{scanner}")
                print(f"[{mfr_name} | ID: {scanner}] (N={m['Count']}): Acc: {m['Accuracy']:.4f} | AUC: {m['ROC_AUC']:.4f}")
                
                breakdown_list.append({
                    "Scanner ID": scanner,
                    "Model": mfr_name,
                    "Count": m['Count'],
                    "Accuracy": m['Accuracy'],
                    "ROC-AUC": m['ROC_AUC']
                })
            
            breakdown_csv = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_scanner_breakdown.csv")
            pd.DataFrame(breakdown_list).to_csv(breakdown_csv, index=False)

        return test_metrics

    def fit(self):
        cm_png_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_confusion_matrix_val.png")
        csv_history_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_training_history.csv")
        metrics_history = []
        
        for epoch in range(self.epochs):
            save_message = ""
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics_history.append({
                "Epoch": epoch + 1, "LR": current_lr,
                "Train_Loss": train_metrics['Loss'], "Val_Loss": val_metrics['Loss'],
                "Train_Acc": train_metrics['Accuracy'], "Val_Acc": val_metrics['Accuracy'],
                "Train_F1": train_metrics['F1'], "Val_F1": val_metrics['F1'],
                "Train_Prec": train_metrics['Precision'], "Val_Prec": val_metrics['Precision'],
                "Train_Sens": train_metrics['Sensitivity'], "Val_Sens": val_metrics['Sensitivity'],
                "Train_Spec": train_metrics['Specificity'], "Val_Spec": val_metrics['Specificity'],
                "Train_AUC": train_metrics['ROC_AUC'], "Val_AUC": val_metrics['ROC_AUC']
            })
            pd.DataFrame(metrics_history).to_csv(csv_history_path, index=False)
            
            if val_metrics['Loss'] < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_metrics['Loss']
                self.epochs_without_improvement = 0
                save_message = "  [Saving: New Best Model!]"
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'model_kwargs': self.model_kwargs
                }
                torch.save(checkpoint, os.path.join(self.save_path, self.save_name))
                
            else:
                self.epochs_without_improvement += 1

            if self.use_wandb and wandb.run is not None:
                wandb.log({
                    "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                    "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                    "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                    "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                    "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC']
                })
            
            print(f"Epoch [{epoch+1}/{self.epochs}] - LR: {current_lr:.6f} | "
                  f"Train Loss: {train_metrics['Loss']:.4f} | Val Loss: {val_metrics['Loss']:.4f} | "
                  f"Train Acc: {train_metrics['Accuracy']:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | "
                  f"Train F1: {train_metrics['F1']:.4f} | Val F1: {val_metrics['F1']:.4f}{save_message}")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping activated at epoch {epoch+1}: no improvement for {self.early_stopping_patience} epochs.")
                break
        
        self.evaluate_best_on_test()
        
        if self.use_wandb and wandb.run is not None:
            wandb.finish()

class KSpaceFlowTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, learning_rate=1e-4, 
                 save_name="best_flow_detector.pth", save_path="runs/kspace_flow", 
                 project_name="K-Print", epochs=100, subfix=[], batch_size=8, model_kwargs=None,
                 early_stopping_patience=15, early_stopping_min_delta=0.001):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_name = save_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.save_path = os.path.join(save_path, "flow_detector", subfix[0] if subfix else "")
        
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0 
        
        config={
            "learning_rate": self.learning_rate,
            "epochs_patience": self.early_stopping_patience,
            "domain": "k-space",
            "batch_size": self.batch_size
        }
        
        if self.model_kwargs:
            config.update(self.model_kwargs)
        
        os.makedirs(self.save_path, exist_ok=True)
        wandb.init(project=project_name, 
                   name=save_name.replace(".pth", ""), 
                   dir=self.save_path, 
                   config=config,
                   tags=["k-space", "flow", "anomaly_detection"] + subfix)
    
    def plot_detection_results(self, labels, scores, filename_prefix="test_detection"):
        """
        Generate and save the plots for Anomaly Detection:
        1. Histogram of Anomaly Score (NLL) separated for Real and Fake.
        2. ROC Curve.
        """
        labels = np.array(labels)
        scores = np.array(scores)
        
        real_scores = scores[labels == 0]
        fake_scores = scores[labels == 1]
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(real_scores, kde=True, color="blue", label="Real MRI (Hardware Signature)", stat="density", alpha=0.5, bins=30)
        sns.histplot(fake_scores, kde=True, color="red", label="Synthetic MRI (Fake)", stat="density", alpha=0.5, bins=30)
        
        plt.title("Distribution of Anomaly Scores (NLL Loss) in k-Space", fontsize=14, fontweight='bold')
        plt.xlabel("Anomaly Score (Negative Log-Likelihood)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        hist_path = os.path.join(self.save_path, f"{filename_prefix}_nll_distribution.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Real Images Misclassified as Fake)', fontsize=12)
        plt.ylabel('True Positive Rate (Fake Images Correctly Detected)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) - K-Space Flow', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        roc_path = os.path.join(self.save_path, f"{filename_prefix}_roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Plots of anomaly detection saved locally in: {self.save_path}")
        
        if wandb.run is not None:
            wandb.log({
                "Test_Set/Plots/NLL_Loss_Distribution": wandb.Image(hist_path),
                "Test_Set/Plots/ROC_Curve_Image": wandb.Image(roc_path)
            })
            
            min_s = np.min(scores)
            max_s = np.max(scores)
            denom = (max_s - min_s) if (max_s - min_s) > 0 else 1.0
            scores_norm = (scores - min_s) / denom
            
            scores_expanded = np.vstack([1.0 - scores_norm, scores_norm]).T
            
            wandb.log({
                "Test_Set/Interactive/ROC_Curve": wandb.plot.roc_curve(
                    y_true=labels, 
                    y_probas=scores_expanded, 
                    labels=["Real", "Fake"]
                )
            })
            print("   - Plots and interactive metrics successfully uploaded to WandB!")
            
    def plot_latent_space(self, dataloader, filename="flow_latent_space.png"):
        print("\n=> Generation of Latent Space Visualization Z (t-SNE & UMAP)...")
        self.model.eval()

        all_z_features = []
        all_labels = []

        with torch.inference_mode():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                targets = batch[1].cpu().numpy()
                
                z, _ = self.model(inputs)
                
                z_energy = z ** 2
                z_compact = F.adaptive_avg_pool3d(z_energy, (4, 4, 4))
                z_flat = z_compact.flatten(1)

                all_z_features.append(z_flat.cpu().numpy())
                all_labels.extend(targets)

        all_z_features = np.concatenate(all_z_features, axis=0)
        all_labels = np.array(all_labels)

        print("   - Calculating t-SNE...")
        perplexity = min(30, len(all_z_features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(all_z_features)

        print("   - Calculating UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_results = reducer.fit_transform(all_z_features)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        palette = {0: "blue", 1: "red"}
        labels_text = ["Real MRI" if lbl == 0 else "Fake MRI" for lbl in all_labels]

        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=labels_text, palette={"Real MRI": "blue", "Fake MRI": "red"}, 
            alpha=0.7, s=80, ax=axes[0], legend=False
        )
        axes[0].set_title("t-SNE (Local structure of Z)", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(
            x=umap_results[:, 0], y=umap_results[:, 1],
            hue=labels_text, palette={"Real MRI": "blue", "Fake MRI": "red"}, 
            alpha=0.7, s=80, ax=axes[1]
        )
        axes[1].set_title("UMAP (Global Topology of Z)", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle("Projection of the Latent Space $Z$ (Flow Energy)", fontsize=16, y=1.02, fontweight='bold')
        plt.tight_layout()

        save_file_path = os.path.join(self.save_path, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Save Latent Space Plot saved in: {save_file_path}")
        
        if wandb.run is not None:
            wandb.log({"Test_Set/Plots/Latent_Space": wandb.Image(save_file_path)})

    def evaluate_best_on_test(self):
        """
        Using the best saved checkpoint, evaluate on the Test Set,
        calculate the metrics and trigger the generation of physical and interactive plots.
        """
        loader_to_use = self.test_loader
        if loader_to_use is None:
            print("\n[Attention] Unable to perform evaluation: no test_loader provided.")
            return None
            
        print("\n" + "="*50 + "\n LOADING BEST MODEL FOR FINAL TEST\n" + "="*50)
        checkpoint_path = os.path.join(self.save_path, self.save_name)
        if not os.path.exists(checkpoint_path):
            print(f"[Error] Checkpoint not found in: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"-> Weights loaded correctly from: {checkpoint_path}")
        
        self.model.eval()
        all_losses = []
        all_labels = []
        
        with torch.inference_mode():
            for batch in loader_to_use:
                inputs = batch[0].to(self.device)
                targets = batch[1].cpu().numpy()
                
                _, loss_vector = self.model(inputs)
                all_losses.extend(loss_vector.tolist())
                all_labels.extend(targets)
        
        try:
            test_auc = roc_auc_score(all_labels, all_losses)
        except ValueError:
            test_auc = float('nan')
            
        mean_loss = np.mean(all_losses)
        
        print("\n" + "="*50 + "\n FINAL RESULTS ON TEST SET (Flow Detector)\n" + "="*50)
        print(f"Test Detection AUC : {test_auc:.4f}")
        print(f"Test Mean NLL Loss : {mean_loss:.4f}")
        print("="*50 + "\n")
        
        print("=> Generating Anomaly Detection plots...")
        self.plot_detection_results(labels=all_labels, scores=all_losses, filename_prefix=self.save_name.replace(".pth", ""))
        
        print("=> Generating Latent Space visualization...")
        self.plot_latent_space(loader_to_use, filename=self.save_name.replace(".pth", "_latent.png"))
        
        if wandb.run is not None:
            wandb.run.summary["Test/Detection_AUC"] = test_auc
            wandb.run.summary["Test/Mean_NLL_Loss"] = mean_loss
            
        return {"Test_AUC": test_auc, "Test_Loss": mean_loss}

    def evaluate(self, dataloader):
        self.model.eval()
        all_losses = []
        all_labels = []
        with torch.inference_mode():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                targets = batch[1].cpu().numpy()  
                
                _, loss_vector = self.model(inputs)
                all_losses.extend(loss_vector.tolist())
                all_labels.extend(targets)
        
        try:
            auc = roc_auc_score(all_labels, all_losses)
        except ValueError:
            auc = float('nan')  
            
        return {"Val_AUC": auc, "Mean_Loss": np.mean(all_losses)}
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            inputs = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            
            _, loss_vector = self.model(inputs)
            
            loss = loss_vector.mean() 
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        return {"Loss": running_loss / len(self.train_loader.dataset)}
    
    def fit(self):
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch()
            
            val_metrics = self.evaluate(self.val_loader)
            
            self.scheduler.step()
            
            train_loss = train_metrics['Loss']
            val_loss = val_metrics['Mean_Loss'] 
            val_auc = val_metrics['Val_AUC']
            
            print(f"Epoch {epoch+1}/{self.epochs} - LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"| Train Loss (NLL): {train_loss:.4f} | Val Loss (NLL): {val_loss:.4f} | Val Detection AUC: {val_auc:.4f}") 
            
            wandb.log({
                "Train_NLL_Loss": train_loss,
                "Val_NLL_Loss": val_loss,
                "Val_Detection_AUC": val_auc
            })
            
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0 
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': float(self.best_val_loss),
                    'model_kwargs': self.model_kwargs
                }
                
                torch.save(checkpoint, os.path.join(self.save_path, self.save_name))
                print("                                                                             -> New Best Model saved! (NLL Loss improved)")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n[!] Early stopping activated at epoch {epoch+1}. The model has not improved for {self.early_stopping_patience} epochs.")
                print(f"Best Val NLL Loss reached: {self.best_val_loss:.4f} (AUC: {val_auc:.4f})")
                break
            
        self.evaluate_best_on_test()
        wandb.finish()

class RealFakeTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=5e-5,
        num_classes=1,
        save_name="best_realfake_model.pth",
        test_loader=None,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        save_path="runs/realfake_experiment",
        project_name="K-Print Real vs Fake",
        epochs=100,
        subfix=[],
        batch_size=8,
        label_smoothing=None,
        model_kwargs=None,
        use_wandb=True,
        folder="",
        isolate_id=None
    ):
        self.use_wandb = use_wandb
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_name = save_name
        self.project_name = project_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0
        self.label_smoothing = label_smoothing
        self.isolate_id = isolate_id

        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.05)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.subfix = subfix
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs or {}
        
        self.best_val_loss = float('inf')  
        self.best_val_metric = 0.0         

        self.generalization = False        
        if self.project_name != "K-Print Real vs Fake":
            self.generalization = True
            self.save_path = os.path.join(save_path, "real_fake_gen", folder, self.subfix[0] if self.subfix else "", self.subfix[2] if len(self.subfix) > 2 else "")
        else:
            self.save_path = os.path.join(save_path, "real_fake", folder, self.subfix[0] if self.subfix else "", self.subfix[2] if len(self.subfix) > 2 else "")

        os.makedirs(self.save_path, exist_ok=True)
        
        config={
            "learning_rate": learning_rate,
            "epochs_patience": early_stopping_patience,
            "num_classes": num_classes,
            "domain": "k-space-binary",
            "batch_size": self.batch_size,
            "loss_function": "BCEWithLogits" if self.num_classes == 1 else "CrossEntropy"
        }
        if self.model_kwargs:
            config.update(self.model_kwargs)
        
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=os.path.splitext(self.save_name)[0].replace("best_", ""),
                group="Real vs Fake Detection",
                dir=self.save_path,
                config=config,
            tags=["binary", "real-fake"] + self.subfix
        )

    def _calculate_metrics(self, targets, preds, probs):
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        precision = precision_score(targets, preds, average='binary', zero_division=0)
        recall = recall_score(targets, preds, average='binary', zero_division=0)
        
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0 
            
        try:
            roc_auc = roc_auc_score(targets, probs)
        except ValueError:
            roc_auc = float('nan')
            
        return acc, f1, precision, recall, specificity, recall, roc_auc, cm

    def _save_confusion_matrix_png(self, cm, save_path, title="Confusion Matrix"):
        """Generate and save the image of the binary confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_latent_space(self, dataloader, filename_binary="latent_space_binary.png", filename_subclasses="latent_space_subclasses.png"):
        print("\n=> Generating Latent Space visualization (t-SNE & UMAP)...")
        
        print("\n=> Generating Latent Space visualization (t-SNE & UMAP)...")
        
        if hasattr(self.model, 'head'):
            original_layer = self.model.head
            self.model.head = nn.Identity()
            layer_name = 'head'
        elif hasattr(self.model, 'fc'):
            original_layer = self.model.fc
            self.model.fc = nn.Identity()
            layer_name = 'fc'
        else:
            print("[Error] Unable to find the final layer ('head' or 'fc') to extract features.")
            return

        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                binary_labels = batch[1].cpu().numpy()
                original_labels = batch[2].cpu().numpy() 
                
                features = self.model(inputs)
                all_features.append(features.cpu().numpy())
                all_binary_labels.append(binary_labels)
                all_original_labels.append(original_labels)

        all_features = np.concatenate(all_features, axis=0)
        all_binary_labels = np.concatenate(all_binary_labels, axis=0).astype(int)
        all_original_labels = np.concatenate(all_original_labels, axis=0)

        print("   - Calculating t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(all_features)

        print("   - Calculating UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_results = reducer.fit_transform(all_features)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        binary_palette = sns.color_palette("Set1", 2)  
        binary_names = np.array(["Real", "Fake"])

        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=binary_names[all_binary_labels], palette=binary_palette, legend=False, alpha=0.8, s=80, ax=axes[0])
        axes[0].set_title("t-SNE (Real vs Fake)", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=binary_names[all_binary_labels], palette=binary_palette, legend="full", alpha=0.8, s=80, ax=axes[1])
        axes[1].set_title("UMAP (Real vs Fake)", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(title="Task Target", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle("Binary Latent Space (Real vs Fake)", fontsize=16, y=1.02, fontweight='bold')
        plt.tight_layout()
        
        save_path_bin = os.path.join(self.save_path, filename_binary)
        plt.savefig(save_path_bin, dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        unique_labels = np.unique(all_original_labels)
        subclass_palette = sns.color_palette("tab20", len(unique_labels))

        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=all_original_labels, palette=subclass_palette, legend=False, alpha=0.8, s=80, ax=axes[0])
        axes[0].set_title("t-SNE (Machinery Subclasses)", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=all_original_labels, palette=subclass_palette, legend="full", alpha=0.8, s=80, ax=axes[1])
        axes[1].set_title("UMAP (Machinery Subclasses)", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(title="Original Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle("Detailed Latent Space (Sub-Classes)", fontsize=16, y=1.02, fontweight='bold')
        plt.tight_layout()
        
        save_path_sub = os.path.join(self.save_path, filename_subclasses)
        plt.savefig(save_path_sub, dpi=300, bbox_inches='tight')
        plt.close()

        if wandb.run is not None:
            wandb.log({
                "Test_Set/Latent_Space_Binary": wandb.Image(save_path_bin),
                "Test_Set/Latent_Space_Subclasses": wandb.Image(save_path_sub)
            })

        print(f"   - Latent Space Visualizations saved in: {self.save_path}")
        
        setattr(self.model, layer_name, original_layer)

    def process_outputs(self, outputs, targets):
        """Uniforms the output regardless of the loss type used and applies Label Smoothing"""
        if self.num_classes == 1:
            outputs = outputs.view(-1)  
            targets_float = targets.float()
            
            if self.label_smoothing is not None and self.label_smoothing > 0:
                targets_float = targets_float * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                
            loss = self.criterion(outputs, targets_float)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
        else:
            loss = self.criterion(outputs, targets.long())
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            preds = torch.argmax(outputs, dim=1)
            
        return loss, preds, probs

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        for batch in self.train_loader:
            inputs = batch[0].to(self.device) 
            targets = batch[1].to(self.device) 
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
    
            loss, preds, probs = self.process_outputs(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
        loss_val = running_loss / len(self.train_loader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
            
        return {"Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec, 
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, "ROC_AUC": auc}

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs, all_original_labels = [], [], [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device) 
                targets = batch[1].to(self.device)
                original_labels = batch[2].cpu().numpy()  
                
                outputs = self.model(inputs)
                loss, preds, probs = self.process_outputs(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_original_labels.extend(original_labels) 
                
        loss_val = running_loss / len(dataloader.dataset)
        acc, f1, prec, rec, spec, sens, auc, cm = self._calculate_metrics(all_targets, all_preds, np.array(all_probs))
        
        return {"Loss": loss_val, "Accuracy": acc, "F1": f1, "Precision": prec,
                "Recall": rec, "Specificity": spec, "Sensitivity": sens, 
                "ROC_AUC": auc, "Confusion_Matrix": cm,
                "Raw_Targets": np.array(all_targets), 
                "Raw_Preds": np.array(all_preds), 
                "Raw_Probs": np.array(all_probs), 
                "Raw_Originals": np.array(all_original_labels)}

    def evaluate_best_on_test(self):
        if self.test_loader is None: return None
        
        test_cm_png_path = os.path.join(self.save_path, f"{self.subfix[0]}_{self.subfix[2]}_confusion_matrix_test.png")
        latent_bin_name = f"{self.subfix[0]}_{self.subfix[2]}_latent_space_binary.png" 
        latent_sub_name = f"{self.subfix[0]}_{self.subfix[2]}_latent_space_subclasses.png" 
        
        checkpoint = torch.load(os.path.join(self.save_path, self.save_name), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader)
        
        print("\n" + "="*50 + "\n Results TEST SET (Best Model Real/Fake)\n" + "="*50)
        print(f"Loss: {test_metrics['Loss']:.4f} | Acc: {test_metrics['Accuracy']:.4f} | "
              f"F1: {test_metrics['F1']:.4f} | Precision: {test_metrics['Precision']:.4f} | "
              f"Recall: {test_metrics['Recall']:.4f} | ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
              f"Spec: {test_metrics['Specificity']:.4f} | Sens: {test_metrics['Sensitivity']:.4f}")
        
        final_cm = test_metrics['Confusion_Matrix']
        self._save_confusion_matrix_png(final_cm, test_cm_png_path, "Confusion Matrix - Test Set (Real vs Fake)")

        label_mapping = {
            0: 'Intera', 1: 'SIGNA EXCITE', 2: 'SIGNA HDx', 3: 'Verio', 
            4: 'Achieva', 5: 'TrioTim', 6: 'MAISI_v2', 
            7: 'Symphony', 8: 'GENESIS_SIGNA', 9: 'MR 7700',
            10: 'SuperSynth_Binary_Segmentation', 11: 'SuperSynth_SynthT1', 
            12: 'SuperSynth_Resampled', 13: 'MED-DDPM'
        }
        
        raw_targets = test_metrics["Raw_Targets"]
        raw_preds = test_metrics["Raw_Preds"]
        raw_probs = test_metrics["Raw_Probs"]
        raw_originals = test_metrics["Raw_Originals"]

        train_real_ids = [0, 1, 2, 3, 4, 5]
        open_set_ids = [7, 8, 9]
        
        train_real_mask = np.isin(raw_originals, train_real_ids)
        open_set_mask = np.isin(raw_originals, open_set_ids)
        fake_mask_global = (raw_targets == 1)

        fake_labels_in_test = np.unique(raw_originals[fake_mask_global])
        fake_results = []
        
        for fake_lbl in fake_labels_in_test:
            fake_name = label_mapping.get(fake_lbl, f"Fake_{fake_lbl}")
            fake_specific_mask = (raw_originals == fake_lbl)
            
            subset_mask = train_real_mask | fake_specific_mask
            sub_targets = raw_targets[subset_mask]
            sub_preds = raw_preds[subset_mask]
            sub_probs = raw_probs[subset_mask]
            
            if len(np.unique(sub_targets)) > 1:
                acc = accuracy_score(sub_targets, sub_preds)
                auc = roc_auc_score(sub_targets, sub_probs)
                avg_prec = average_precision_score(sub_targets, sub_probs)
                isolated_fake_acc = accuracy_score(raw_targets[fake_specific_mask], raw_preds[fake_specific_mask])
                
                fake_results.append({
                    "Task": f"Train Reals vs {fake_name}",
                    "Isolated Fake/Real Acc.": isolated_fake_acc,
                    "Task Accuracy": acc,
                    "PR-AUC": avg_prec,
                    "ROC-AUC": auc
                })
        
        subset_mask_all = train_real_mask | fake_mask_global
        sub_targets_all = raw_targets[subset_mask_all]
        sub_preds_all = raw_preds[subset_mask_all]
        sub_probs_all = raw_probs[subset_mask_all]
        
        if len(np.unique(sub_targets_all)) > 1:
            isolated_fake_acc_all = accuracy_score(raw_targets[fake_mask_global], raw_preds[fake_mask_global])
            fake_results.append({
                "Task": "ALL Train Reals vs ALL Fakes (MICRO-AVERAGE)",
                "Isolated Fake/Real Acc.": isolated_fake_acc_all,
                "Task Accuracy": accuracy_score(sub_targets_all, sub_preds_all),
                "PR-AUC": average_precision_score(sub_targets_all, sub_probs_all),
                "ROC-AUC": roc_auc_score(sub_targets_all, sub_probs_all)
            })

        open_set_results = []
        
        subset_mask_open = open_set_mask | fake_mask_global
        sub_targets_open = raw_targets[subset_mask_open]
        sub_preds_open = raw_preds[subset_mask_open]
        sub_probs_open = raw_probs[subset_mask_open]
        
        if len(np.unique(sub_targets_open)) > 1:
            isolated_real_acc = accuracy_score(raw_targets[open_set_mask], raw_preds[open_set_mask])
            open_set_results.append({
                "Task": "ALL Open Set vs ALL Fakes",
                "Isolated Real Acc.": isolated_real_acc,
                "Task Accuracy": accuracy_score(sub_targets_open, sub_preds_open),
                "PR-AUC": average_precision_score(sub_targets_open, sub_probs_open),
                "ROC-AUC": roc_auc_score(sub_targets_open, sub_probs_open)
            })

        isolate_results = []
        
        if self.isolate_id is not None:
            isolate_mask = np.isin(raw_originals, self.isolate_id)
            subset_mask_isolate = isolate_mask | fake_mask_global
            
            sub_targets_iso = raw_targets[subset_mask_isolate]
            sub_preds_iso = raw_preds[subset_mask_isolate]
            sub_probs_iso = raw_probs[subset_mask_isolate]
            
            if len(np.unique(sub_targets_iso)) > 1:
                isolated_real_acc_iso = accuracy_score(raw_targets[isolate_mask], raw_preds[isolate_mask])
                isolate_results.append({
                    "Task": f"ALL Isolated Reals vs ALL Fakes",
                    "Isolated Real Acc.": isolated_real_acc_iso,
                    "Task Accuracy": accuracy_score(sub_targets_iso, sub_preds_iso),
                    "PR-AUC": average_precision_score(sub_targets_iso, sub_probs_iso),
                    "ROC-AUC": roc_auc_score(sub_targets_iso, sub_probs_iso)
                })

        manufacturers_ids = {
            'GE': [1, 2],
            'SIEMENS': [3, 5],
            'PHILIPS': [0, 4]
        }

        model_name = "All" 
        if self.isolate_id is not None:
            target_id = self.isolate_id[0] 
            
            for manufacturer, ids in manufacturers_ids.items():
                if target_id in ids:
                    model_name = manufacturer
                    break
        self._process_and_log_table(fake_results, f"{self.subfix[0]}_{self.subfix[2]}_detailed_{model_name}_fake_metrics.csv", "Detailed Fake Metrics", "Detailed_Fake_Metrics", calc_mean=False)
        self._process_and_log_table(open_set_results, f"{self.subfix[0]}_{self.subfix[2]}_detailed_{model_name}_open_set_metrics.csv", "Detailed Open Set Metrics", "Detailed_Open_Set_Metrics")
        if self.isolate_id is not None and len(isolate_results) > 0:
            self._process_and_log_table(
                isolate_results, 
                f"{self.subfix[0]}_{self.subfix[2]}_detailed_{model_name}_isolate_manufactor_metrics.csv", 
                "REPORT DETTAGLIATO: MACCHINARI ISOLATI VS ALL FAKES", 
                "Detailed_Isolate_Metrics"
            )

        if wandb.run is not None:
            wandb.run.summary["Test/Loss"] = test_metrics['Loss']
            wandb.run.summary["Test/Accuracy"] = test_metrics['Accuracy']
            wandb.run.summary["Test/f1"] = test_metrics['F1']
            wandb.run.summary["Test/precision"] = test_metrics['Precision']
            wandb.run.summary["Test/recall"] = test_metrics['Recall']
            wandb.run.summary["Test/specificity"] = test_metrics['Specificity']
            wandb.run.summary["Test/sensitivity"] = test_metrics['Sensitivity']
            wandb.run.summary["Test/ROC_AUC"] = test_metrics['ROC_AUC']
            wandb.log({"Test/Confusion_Matrix": wandb.Image(test_cm_png_path)})
        
        return test_metrics

    def _process_and_log_table(self, results_list, csv_name, title, wandb_name, calc_mean=True):
        if results_list:
            df = pd.DataFrame(results_list)
            
            if calc_mean:
                mean_row = pd.DataFrame([{
                    "Task": "MEAN (Media Globale)",
                    "Isolated Fake Acc.": df["Isolated Fake Acc."].mean() if "Isolated Fake Acc." in df else None,
                    "Isolated Real Acc.": df["Isolated Real Acc."].mean() if "Isolated Real Acc." in df else None,
                    "Task Accuracy": df["Task Accuracy"].mean(),
                    "PR-AUC": df["PR-AUC"].mean(),
                    "ROC-AUC": df["ROC-AUC"].mean()
                }]).dropna(axis=1, how='all')
                
                df = pd.concat([df, mean_row], ignore_index=True)
            
            df = df.round(2)
            df.to_csv(os.path.join(self.save_path, csv_name), index=False)
            
            print(f"\n{'-'*75}\n {title}\n{'-'*75}")
            print(df.to_string(index=False))
            print("-" * 75)
            
            if wandb.run is not None:
                wandb.log({f"Test/{wandb_name}": wandb.Table(dataframe=df)})

    def fit(self, epochs=None):
        if epochs is not None: self.epochs = epochs
        csv_history_path = os.path.join(self.save_path, f"{os.path.splitext(self.save_name)[0]}_training_history.csv")
        metrics_history = []
        
        for epoch in range(self.epochs):
            save_message = ""
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics_history.append({
                "Epoch": epoch + 1, "LR": current_lr,
                "Train_Loss": train_metrics['Loss'], "Val_Loss": val_metrics['Loss'],
                "Train_Acc": train_metrics['Accuracy'], "Val_Acc": val_metrics['Accuracy'],
                "Train_F1": train_metrics['F1'], "Val_F1": val_metrics['F1'],
                "Train_AUC": train_metrics['ROC_AUC'], "Val_AUC": val_metrics['ROC_AUC']
            })
            pd.DataFrame(metrics_history).to_csv(csv_history_path, index=False)
            
            if val_metrics['Loss'] < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_metrics['Loss']
                save_message = "  [Salvataggio: Nuovo Best Model!]"
                self.epochs_without_improvement = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'model_kwargs': self.model_kwargs
                }
                torch.save(checkpoint, os.path.join(self.save_path, self.save_name))
            else:
                self.epochs_without_improvement += 1

            wandb.log({
                "Epoch": epoch + 1, "Hyperparameters/LR": current_lr,
                "Loss/Train": train_metrics['Loss'], "Loss/Validation": val_metrics['Loss'],
                "Accuracy/Train": train_metrics['Accuracy'], "Accuracy/Validation": val_metrics['Accuracy'],
                "F1_Score/Train": train_metrics['F1'], "F1_Score/Validation": val_metrics['F1'],
                "ROC_AUC/Train": train_metrics['ROC_AUC'], "ROC_AUC/Validation": val_metrics['ROC_AUC']
            })
                
            print(f"Epoch [{epoch+1}/{self.epochs}] - LR: {current_lr:.6f} | "
                  f"Train Loss: {train_metrics['Loss']:.4f} | Val Loss: {val_metrics['Loss']:.4f} | "
                  f"Train Acc: {train_metrics['Accuracy']:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | "
                  f"Train F1: {train_metrics['F1']:.4f} | Val F1: {val_metrics['F1']:.4f}{save_message}")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping attivato all'epoca {epoch+1}")
                break
        
        self.evaluate_best_on_test()
        wandb.finish()