import torch
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    RandFlip,
    RandGaussianNoise,
    RandRotate,
    ScaleIntensityRangePercentiles, 
    NormalizeIntensity
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
PURE_DATASET_ROOT = PROJECT_ROOT / "dataset"
GLOBAL_SPECTRUMS_FOLDER = PURE_DATASET_ROOT / "kspace_global_spectrums"
COMB_SPECTRUMS_FOLDER = PURE_DATASET_ROOT / "kspace_double_split_spectrums"
SINGLE_SPECTRUMS_FOLDER = PURE_DATASET_ROOT / "kspace_global_single_spectrums"
MANUFACTURER_SPECTRUMS_FOLDER = PURE_DATASET_ROOT / "kspace_manufacture_spectrums"

DATASET_RAW = CURRENT_FILE.parents[3] / "dataset"

warnings.filterwarnings("ignore", category=UserWarning)

class KPrintDataset(Dataset):
    def __init__(self, data, target_shape=(128, 190, 190), augment=False, use_registered=True, normalization=True, pure_ablation=False, gender=False, ablation_type="Pure_KSS"):
        """
        Dataset per K-Print: carica dal CSV e applica la pipeline MONAI nativa
        per garantire la conservazione dei metadati fisici e l'assenza di distorsioni.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True) 
        self.use_registered = use_registered
        self.normalization = normalization
        self.pure_ablation = pure_ablation 
        self.gender = gender
        self.ablation_type = ablation_type
        
        if self.pure_ablation:
            print(f"⚠️  Modalità Pure Ablation attivata: {self.ablation_type}. I dati saranno caricati da {PURE_DATASET_ROOT}/{self.ablation_type}.")

        transforms = [
            LoadImage(image_only=True),                     # 1. Carica il NIfTI preservando l'affine matrix
            EnsureChannelFirst(channel_dim="no_channel"),   # 2. Aggiunge il canale [1, W, H, D]
            Orientation(axcodes="RAS"),                     # 3. Riorientamento anatomico sicuro
        ]

        if not self.use_registered:
            transforms.append(
                CenterSpatialCrop(roi_size=target_shape)    # 4. Crop centrale senza padding solo per MRI raw
            )
        
        if self.normalization:
            transforms.extend([
                ScaleIntensityRangePercentiles(
                    lower=1.0, 
                    upper=99.0, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True, 
                    relative=False
                ),
                NormalizeIntensity(nonzero=True, channel_wise=True)
            ])

        if augment:
            transforms.extend([
                # RandRotate(
                #     range_x=np.pi / 18,
                #     range_y=np.pi / 18,
                #     range_z=np.pi / 18,
                #     prob=0.5,
                #     padding_mode="border"
                # ),
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
                # RandGaussianNoise(prob=0.5, mean=0.0, std=0.01)
            ])

        # Pipeline MONAI unificata: fa tutto lei a partire dal path del file!
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        project = row['Project']
        prefix = row['prefix']
        label = row['Model']
        
        # Costruzione del path dinamico
        if self.pure_ablation: 
            KSS_dataset = "dataset_without_KSS" if self.ablation_type == "Pure_KSS" else "dataset_without_KSS_Control"
            file_path = f"{PURE_DATASET_ROOT}/{KSS_dataset}/{prefix}.nii.gz"
        elif project == 'ADNI':
            folder = "ADNI_registered" if self.use_registered else "ADNI_nifti"
            file_path = f"{DATASET_RAW}/ADNI/{folder}/{prefix}.nii.gz"
        elif project == 'PPMI':
            folder = "PPMI_registered" if self.use_registered else "PPMI_nifti"
            file_path = f"{DATASET_RAW}/PPMI/{folder}/{prefix}.nii.gz"
        else:
            raise ValueError(f"Progetto non riconosciuto: {project}")
            
        # Passiamo direttamente il percorso del file alla trasformazione MONAI.
        # Restituirà il tensore spaziale V(x,y,z) già elaborato e perfetto per la rete.
        V_tensor = self.transform(file_path).float()

        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.gender: 
            gender = row['Sex']
            gender_tensor = torch.tensor(gender, dtype=torch.long)
            return V_tensor, label_tensor, gender_tensor
        else:
            return V_tensor, label_tensor
        
class KSpace_Dataset(Dataset):
    def __init__(self, data, global_spectrum_mode="GLOBAL", pure_ablation=False, gender=False, ablation_type="Pure_KSS", cropping = False, cropping_shape=(120, 120, 120)):
        """
        Dataset per K-Print: carica dal CSV e applica la pipeline MONAI nativa
        per garantire la conservazione dei metadati fisici e l'assenza di distorsioni.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True) 
        self.pure_ablation = pure_ablation 
        self.gender = gender
        self.ablation_type = ablation_type
        self.global_spectrum_mode = global_spectrum_mode
        self.cropping = cropping
        self.cropping_shape = cropping_shape
        # SALVA IL MAPPING QUI: il dataset "ricorderà" chi sono le classi
        self.label_mapping = dict(zip(self.df['Model'], self.df['Mfg Model']))
        
        if self.pure_ablation:
            print(f"⚠️  Modalità Pure Ablation attivata: {self.ablation_type}. I dati saranno caricati da {PURE_DATASET_ROOT}/{self.ablation_type}.")

        if self.cropping:
            print(f"⚠️  Modalità Cropping attivata: i volumi saranno centrati e croppati a {self.cropping_shape}.")
            self.transform = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(channel_dim="no_channel"),
                CenterSpatialCrop(roi_size=self.cropping_shape)
            ])
        else:
            self.transform = Compose([
            LoadImage(image_only=True),                     
            EnsureChannelFirst(channel_dim="no_channel")   
        ])

        # --- PRE-CARICAMENTO DEGLI SPETTRI IN RAM ---
        self.global_spectrum_tensor = None
        self.class_spectrums_tensors = {}

        # 1. Modalità con singolo spettro globale
        if self.global_spectrum_mode in ["GLOBAL", "SPLIT1", "SPLIT2"]:
            spectrum_path = GLOBAL_SPECTRUMS_FOLDER / f"mean_kspace_{self.global_spectrum_mode}.nii.gz"
            self.global_spectrum_tensor = self.transform(spectrum_path).float()
            
        elif "COMB" in self.global_spectrum_mode:
            spectrum_path = COMB_SPECTRUMS_FOLDER / f"mean_kspace_{self.global_spectrum_mode}.nii.gz"
            self.global_spectrum_tensor = self.transform(spectrum_path).float()
            
        elif "Model" in self.global_spectrum_mode:
            spectrum_path = SINGLE_SPECTRUMS_FOLDER / f"mean_kspace_{self.global_spectrum_mode}.nii.gz"
            self.global_spectrum_tensor = self.transform(spectrum_path).float()
            
        elif self.global_spectrum_mode == "Detector_Real" or "Anomaly" in self.global_spectrum_mode or self.global_spectrum_mode == "REAL_TEST":
            if "Anomaly_" in self.global_spectrum_mode:
                spectrum_name = self.global_spectrum_mode.split("Anomaly_")[1]  # Estrae la combinazione specifica (es. "0_1")
                spectrum_path = COMB_SPECTRUMS_FOLDER / f"mean_kspace_COMB_{spectrum_name}.nii.gz"
                self.global_spectrum_tensor = self.transform(spectrum_path).float()
            else:
                spectrum_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL.nii.gz"
                self.global_spectrum_tensor = self.transform(spectrum_path).float()
            
        elif self.global_spectrum_mode == "Detector_Synth":
            spectrum_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL_Synth.nii.gz"
            self.global_spectrum_tensor = self.transform(spectrum_path).float()
            
        elif self.global_spectrum_mode == "Detector_Mean":
            real_tensor = self.transform(GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL.nii.gz").float()
            synth_tensor = self.transform(GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL_Synth.nii.gz").float()
            self.global_spectrum_tensor = (real_tensor + synth_tensor) / 2.0

        # 2. Modalità dipendenti dalla classe (Spettri multipli)
        elif self.global_spectrum_mode == "Detector":
            real_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL.nii.gz"
            synth_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL_Synth.nii.gz"
            self.class_spectrums_tensors[0] = self.transform(real_path).float()   # Reale
            self.class_spectrums_tensors[1] = self.transform(synth_path).float()  # Sintetica
            
        elif self.global_spectrum_mode == "Scanner":
            print("Caricamento in memoria degli spettri di classe...")
            unique_labels = self.df['Model'].unique()
            for lbl in unique_labels:
                spectrum_path = GLOBAL_SPECTRUMS_FOLDER / f"mean_kspace_Model_{lbl}.nii.gz"
                if not spectrum_path.exists():
                    raise FileNotFoundError(f"Spettro di classe mancante: {spectrum_path}")
                self.class_spectrums_tensors[lbl] = self.transform(spectrum_path).float()
        
        else:
            print(f"Modalità global_spectrum_mode '{self.global_spectrum_mode}' non riconosciuta. Nessuna normalizzazione globale sarà applicata.")
            choices=['GLOBAL', 'Scanner', 'SPLIT1', 'SPLIT2',
                    "COMB_0_1", "COMB_0_2", "COMB_0_3", "COMB_0_4", "COMB_0_5",
                    "COMB_1_2", "COMB_1_3", "COMB_1_4", "COMB_1_5",
                    "COMB_2_3", "COMB_2_4", "COMB_2_5",
                    "COMB_3_4", "COMB_3_5",
                    "COMB_4_5",
                    "Model_0", "Model_1", "Model_2", "Model_3", "Model_4", "Model_5",
                    "Detector", "Detector_Real", "Detector_Synth", "Detector_Mean",
                    "Anomaly", "Anomaly_{ COMB_x_y }", "REAL_TEST"]
            print(f"Scelte valide per global_spectrum_mode: {choices}")
        
        print(f"Modalità global_spectrum_mode: {self.global_spectrum_mode}")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row['prefix']
        original_label = row['Model']
        
        # Gestione del task: se siamo in modalità Detector, il target diventa binario
        is_detector_task = "Detector" in self.global_spectrum_mode or "Anomaly" in self.global_spectrum_mode or "REAL_TEST" in self.global_spectrum_mode

        if is_detector_task:
            label = 1 if original_label == 6 else 0 # 1 = Sintetica (Anomala), 0 = Reale (Normale)
        else:
            label = original_label
        
        if self.pure_ablation:
            file_path = PURE_DATASET_ROOT / self.ablation_type / f"{prefix}.nii.gz"
        else:
            file_path = PURE_DATASET_ROOT / "kspace" / f"{prefix}.nii.gz"
       
        try:
            V_tensor = self.transform(file_path).float()
        except Exception as e:
            print(f"Errore nel caricamento del file {file_path}: {e}")
            raise

        # --- SOTTRAZIONE SPETTRO ---
        if self.global_spectrum_tensor is not None:
            # Cattura GLOBAL, SPLIT, COMB, Model, Detector_Real, Detector_Synth, Detector_Mean
            V_tensor -= self.global_spectrum_tensor.to(V_tensor.device)
            
        elif self.global_spectrum_mode == "Detector":
            # Dipende se l'immagine è reale (0) o sintetica (1)
            V_tensor -= self.class_spectrums_tensors[label].to(V_tensor.device)
            
        elif self.global_spectrum_mode == "Scanner":
            # Scanner usa lo spettro in base al modello originale, non quello binarizzato
            V_tensor -= self.class_spectrums_tensors[original_label].to(V_tensor.device)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.gender: 
            gender_tensor = torch.tensor(row['Sex'], dtype=torch.long)
            return V_tensor, label_tensor, gender_tensor
        
        if "REAL_TEST" in self.global_spectrum_mode:
            return V_tensor, label_tensor, prefix  # Ritorna anche il prefix per identificare i campioni reali durante la valutazione
        
        return V_tensor, label_tensor

class FangDataset(Dataset):
    def __init__(self, data, patch_size=128, use_registered=True):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True) 
            
        self.patch_size = patch_size
        self.use_registered = use_registered
        
        # Carichiamo il volume 3D, poi estrarremo la fetta in __getitem__
        self.transform = Compose([
            LoadImage(image_only=True),                     
            EnsureChannelFirst(channel_dim="no_channel"),   
            Orientation(axcodes="RAS")                     
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        project = row['Project']
        prefix = row['prefix']
        label = row['Model']
        
        if project == 'ADNI':
            folder = "ADNI_registered" if self.use_registered else "ADNI_nifti"
            file_path = f"../../../dataset/ADNI/{folder}/{prefix}.nii.gz"
        elif project == 'PPMI':
            folder = "PPMI_registered" if self.use_registered else "PPMI_nifti"
            file_path = f"../../../dataset/PPMI/{folder}/{prefix}.nii.gz"
        else:
            raise ValueError(f"Progetto non riconosciuto: {project}")
            
        # 1. Carica il volume intero [1, W, H, D]
        V_tensor = self.transform(file_path).float()
        
        # 2. Estrazione Middle Slice (asse D)
        D = V_tensor.shape[3]
        middle_idx = D // 2
        slice_2d = V_tensor[:, :, :, middle_idx] # Diventa [1, W, H]
        
        # 3. Center Crop a 128x128 manuale per la fetta 2D
        _, W, H = slice_2d.shape
        start_w = max(0, (W - self.patch_size) // 2)
        start_h = max(0, (H - self.patch_size) // 2)
        
        patch_2d = slice_2d[:, start_w:start_w+self.patch_size, start_h:start_h+self.patch_size]
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_2d, label_tensor

class RealFake_KSpace_Dataset(Dataset):
    def __init__(self, data, use_global_subtraction=True, kspace=True, ablation_type="Pure_KSS", generalization=None):
        """
        Simplified Dataset for Binary Classification (Real vs Fake).
        Allows optional subtraction of a single GLOBAL spectrum.
        """
        # 1. Load Data
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True) 
            
        self.ablation_type = ablation_type
        self.use_global_subtraction = use_global_subtraction
        self.kspace = kspace
        self.generalization = generalization
        
        self.label_mapping = {
            0: 'Intera', 
            1: 'SIGNA EXCITE', 
            2: 'SIGNA HDx', 
            3: 'Verio', 
            4: 'Achieva', 
            5: 'TrioTim', 
            6: 'MAISI_v2', 
            7: 'Symphony',
            8: 'GENESIS_SIGNA',
            9: 'MR 7700',
            10: 'SuperSynth_Binary_Segmentation', 
            11: 'SuperSynth_SynthT1', 
            12: 'SuperSynth_Resampled', 
            13: 'MED-DDPM'
        }

        # 2. Configure Transforms
        transforms_list = [
            LoadImage(image_only=True),                     
            EnsureChannelFirst(channel_dim="no_channel")   
        ]
            
        self.transform = Compose(transforms_list)

        # 3. Pre-load Global Spectrum
        self.global_spectrum_tensor = None
        
        if self.use_global_subtraction and self.kspace:
            
            if not self.generalization:
                spectrum_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL.nii.gz"
                print(f"Warning: Global Subtraction enabled: loading spectrum from {spectrum_path}")
            
            else:
                mapping_path = {
                    "GE": "mean_kspace_COMB_1_2.nii.gz",
                    "SIEMENS": "mean_kspace_COMB_3_5.nii.gz",
                    "PHILIPS": "mean_kspace_COMB_0_4.nii.gz"
                }
                
                filename = mapping_path.get(self.generalization)
                if filename:
                    spectrum_path = COMB_SPECTRUMS_FOLDER / filename
                    print(f"DEBUG: Caricamento spettro SPECIFICO per {self.generalization}: {spectrum_path}")
                else:
                    raise ValueError(f"Generalization mode '{self.generalization}' non mappata.")
            
            try:
                self.global_spectrum_tensor = self.transform(spectrum_path).float()
            except Exception as e:
                print(f"Error loading global spectrum: {e}")
                raise
        else:
            print("Warning: Global Subtraction disabled: using raw images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row['prefix']
        original_label = row['Model']
        
        # Binary Task logic: 1 for Fake (assuming Model >= 6 is fake), 0 for Real
        binary_label = 1 if original_label == 6 or original_label >= 10 else 0
        
        if self.kspace:
            # Construct File Path
            file_path = PURE_DATASET_ROOT / "kspace" / f"{prefix}.nii.gz"

        else:
            folder = row['Project']

            if pd.isna(folder):

                model_name = row["Mfg Model"]
                if "Super" in model_name:
                    folder = "SuperSynth/"

                    split = prefix.split("_")
                
                    folder += f"{split[0]}_{split[1]}_{split[2]}_{split[3]}" if "_S_" in prefix else f"{split[0]}_{split[1]}" 

                elif "MAISI" in model_name:
                    folder = "NV-Generate-CTMR"
                elif "MED-DDPM" in model_name:
                    folder = "MED-DDPM"
                elif original_label in [7, 8, 9]:
                    folder = "ADNI/ADNI_registered"
                else:
                    raise ValueError(f"Unknown project for prefix {prefix} with label {original_label}")
                
                file_path = PURE_DATASET_ROOT / "registered" / folder / f"{prefix}.nii.gz"

            else:
                file_path = PURE_DATASET_ROOT / "registered" / folder / f"{folder}_registered" / f"{prefix}.nii.gz"

        # Load Volume
        try:
            V_tensor = self.transform(file_path).float()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise

        # Apply Global Subtraction if the tensor exists
        if self.global_spectrum_tensor is not None:
            V_tensor -= self.global_spectrum_tensor.to(V_tensor.device)
            
        # label_tensor uses float32 (standard for binary cross entropy)
        label_tensor = torch.tensor(binary_label, dtype=torch.float32) 
        
        # original_label_tensor uses long (standard for integer IDs)
        original_label_tensor = torch.tensor(original_label, dtype=torch.long) 
        
        return V_tensor, label_tensor, original_label_tensor
    
class Gender_KSpace_Dataset(Dataset):
    def __init__(self, data, split_folder_name=None, KSS=False):
        """
        Dataset per la Classificazione del Genere.
        Gestisce l'armonizzazione dinamica (KSS) per il Multi-Source Domain Adaptation.
        """
        self.df = data.copy().reset_index(drop=True) if not isinstance(data, str) else pd.read_csv(data)

        self.KSS = KSS

        # --- CENTRALIZZAZIONE DEL MAPPING ---
        # Mappiamo M->0 e F->1 per tutta la colonna. 
        # Inseriamo anche 0 e 1 nel dizionario per sicurezza, casomai arrivasse un DF già mappato
        mapping = {'M': 0, 'F': 1, 'm': 0, 'f': 1, 0: 0, 1: 1, '0': 0, '1': 1}
        if 'Sex' in self.df.columns:
            self.df['Sex'] = self.df['Sex'].map(mapping)
            
            # Controllo di sicurezza rapido
            if self.df['Sex'].isnull().any():
                print("⚠️ Attenzione: Trovati valori nulli o non riconosciuti nella colonna Sex dopo il mapping!")
        # ------------------------------------

        self.model_to_mfr = {
            1: 'GE', 2: 'GE',
            3: 'SIEMENS', 5: 'SIEMENS',
            0: 'PHILIPS', 4: 'PHILIPS'
        }

        transforms_list = [
            LoadImage(image_only=True),                     
            EnsureChannelFirst(channel_dim="no_channel")   
        ]
        self.transform = Compose(transforms_list)

        self.kss_tensors = {}
        
        # Gestione Cartella KSS e Pre-caricamento
        if self.KSS and split_folder_name:
            split_kss_dir = MANUFACTURER_SPECTRUMS_FOLDER / split_folder_name
            print(f"[*] Inizializzazione Armonizzazione KSS. Lettura da: {split_kss_dir.name}")
            
            for mfr in ['GE', 'SIEMENS', 'PHILIPS']:
                kss_path = split_kss_dir / f"KSS_{mfr}.nii.gz"
                if kss_path.exists():
                    try:
                        self.kss_tensors[mfr] = torch.from_numpy(nib.load(kss_path).get_fdata(dtype=np.float32))
                        print(f"    -> Caricato KSS per {mfr}")
                    except Exception as e:
                        print(f"    -> Errore nel caricamento del KSS per {mfr}: {e}")
                else:
                    print(f"    -> Nessun KSS trovato per {mfr} (Corretto se è il dominio di Test Unseen)")
        elif self.KSS:
            print("[!] Attenzione: KSS richiesto ma nessuno 'split_folder_name' fornito.")
        else:
            print("[*] Armonizzazione KSS disabilitata. Uso delle immagini Raw.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row['prefix']
        model_id = row['Model']
        mfr_name = self.model_to_mfr.get(model_id, 'UNKNOWN')
        gender_label = torch.tensor(row['Sex'], dtype=torch.float32)
        
        # SEZIONE ARMONIZZAZIONE (KSS = True)
        if self.KSS:
            # Carica file dal dataset K-Space (già trasformato in frequenza)
            file_path = PROJECT_ROOT / "dataset" / "kspace_amp_phase" / f"{prefix}.nii.gz"
            MRI_f = torch.from_numpy(nib.load(file_path).get_fdata(dtype=np.complex64))
            
            # Sottrazione firma hardware (KSS)
            if mfr_name in self.kss_tensors:
                magnitude = torch.abs(MRI_f)
                phase = torch.angle(MRI_f)

                MRI_raw = torch.log(1 + magnitude**2)
                MRI_filterd = MRI_raw - self.kss_tensors[mfr_name]

                magnitude_filtered = torch.sqrt(torch.clamp(torch.exp(MRI_filterd) - 1, min=0.0))
                S_filtered_complex = magnitude_filtered * torch.exp(1j * phase)

            else:
                print(f"[!] Nessun KSS trovato per {mfr_name}. Nessuna sottrazione applicata.")
                S_filtered_complex = MRI_f

            # ricostruzione dello spazio reale (IFFT)
            MRI_f_shifted = torch.fft.ifftshift(S_filtered_complex, dim=(-3, -2, -1))
            MRI_fitered = torch.fft.ifftn(MRI_f_shifted, dim=(-3, -2, -1), norm="ortho").real.float().unsqueeze(0) 

            MRI_f = MRI_fitered.real

        # SEZIONE RAW (KSS = False)
        else:
            # Carica file dallo spazio reale (Registered)
            sub_dir = "ADNI/ADNI_registered" if row['Project'] == 'ADNI' else "PPMI/PPMI_registered"
            file_path = PROJECT_ROOT / "dataset" / "registered" / sub_dir / f"{prefix}.nii.gz"
            MRI_f = self.transform(file_path).float() 

        return MRI_f, gender_label, torch.tensor(row['Model'], dtype=torch.long)