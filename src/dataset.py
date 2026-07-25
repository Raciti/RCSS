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
        Dataset for K-Print: loads from CSV and applies the native MONAI pipeline
        to ensure the preservation of physical metadata and the absence of distortions.
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
            print(f"⚠️  Pure Ablation mode activated: {self.ablation_type}. Data will be loaded from {PURE_DATASET_ROOT}/{self.ablation_type}.")

        transforms = [
            LoadImage(image_only=True),                     # 1. Loads the NIfTI while preserving the affine matrix
            EnsureChannelFirst(channel_dim="no_channel"),   # 2. Adds the channel [1, W, H, D]
            Orientation(axcodes="RAS"),                     # 3. Safe anatomical reorientation
        ]

        if not self.use_registered:
            transforms.append(
                CenterSpatialCrop(roi_size=target_shape)    # 4. Central crop without padding only for raw MRI
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
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
            ])

        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        project = row['Project']
        prefix = row['prefix']
        label = row['Model']
        
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
        Dataset for K-Print: loads from CSV and applies the native MONAI pipeline
        to ensure the preservation of physical metadata and the absence of distortions.
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

        self.label_mapping = dict(zip(self.df['Model'], self.df['Mfg Model']))
        
        if self.pure_ablation:
            print(f"⚠️  Pure Ablation mode activated: {self.ablation_type}. Data will be loaded from {PURE_DATASET_ROOT}/{self.ablation_type}.")

        if self.cropping:
            print(f"⚠️  Cropping mode activated: volumes will be centered and cropped to {self.cropping_shape}.")
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

        self.global_spectrum_tensor = None
        self.class_spectrums_tensors = {}

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
                spectrum_name = self.global_spectrum_mode.split("Anomaly_")[1] 
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

        elif self.global_spectrum_mode == "Detector":
            real_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL.nii.gz"
            synth_path = GLOBAL_SPECTRUMS_FOLDER / "mean_kspace_GLOBAL_Synth.nii.gz"
            self.class_spectrums_tensors[0] = self.transform(real_path).float()  
            self.class_spectrums_tensors[1] = self.transform(synth_path).float() 
            
        elif self.global_spectrum_mode == "Scanner":
            print("Loading class spectra into memory...")
            unique_labels = self.df['Model'].unique()
            for lbl in unique_labels:
                spectrum_path = GLOBAL_SPECTRUMS_FOLDER / f"mean_kspace_Model_{lbl}.nii.gz"
                if not spectrum_path.exists():
                    raise FileNotFoundError(f"Spettro di classe mancante: {spectrum_path}")
                self.class_spectrums_tensors[lbl] = self.transform(spectrum_path).float()
        
        else:
            print(f"Global spectrum mode '{self.global_spectrum_mode}' not recognized. No global normalization will be applied.")
            choices=['GLOBAL', 'Scanner', 'SPLIT1', 'SPLIT2',
                    "COMB_0_1", "COMB_0_2", "COMB_0_3", "COMB_0_4", "COMB_0_5",
                    "COMB_1_2", "COMB_1_3", "COMB_1_4", "COMB_1_5",
                    "COMB_2_3", "COMB_2_4", "COMB_2_5",
                    "COMB_3_4", "COMB_3_5",
                    "COMB_4_5",
                    "Model_0", "Model_1", "Model_2", "Model_3", "Model_4", "Model_5",
                    "Detector", "Detector_Real", "Detector_Synth", "Detector_Mean",
                    "Anomaly", "Anomaly_{ COMB_x_y }", "REAL_TEST"]
            print(f"Valid choices for global_spectrum_mode: {choices}")
        
        print(f"Global spectrum mode: {self.global_spectrum_mode}")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row['prefix']
        original_label = row['Model']
        
        is_detector_task = "Detector" in self.global_spectrum_mode or "Anomaly" in self.global_spectrum_mode or "REAL_TEST" in self.global_spectrum_mode

        if is_detector_task:
            label = 1 if original_label == 6 else 0  
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

        if self.global_spectrum_tensor is not None:
            V_tensor -= self.global_spectrum_tensor.to(V_tensor.device)
            
        elif self.global_spectrum_mode == "Detector":
            V_tensor -= self.class_spectrums_tensors[label].to(V_tensor.device)
            
        elif self.global_spectrum_mode == "Scanner":
            V_tensor -= self.class_spectrums_tensors[original_label].to(V_tensor.device)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.gender: 
            gender_tensor = torch.tensor(row['Sex'], dtype=torch.long)
            return V_tensor, label_tensor, gender_tensor
        
        if "REAL_TEST" in self.global_spectrum_mode:
            return V_tensor, label_tensor, prefix  
        
        return V_tensor, label_tensor

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
        Gender Classification Dataset.
        Manages dynamic harmonization (KSS) for Multi-Source Domain Adaptation.
        """
        self.df = data.copy().reset_index(drop=True) if not isinstance(data, str) else pd.read_csv(data)

        self.KSS = KSS

        mapping = {'M': 0, 'F': 1, 'm': 0, 'f': 1, 0: 0, 1: 1, '0': 0, '1': 1}
        if 'Sex' in self.df.columns:
            self.df['Sex'] = self.df['Sex'].map(mapping)
            
            if self.df['Sex'].isnull().any():
                print("⚠️ Warning: Null or unrecognized values found in the Sex column after mapping!")
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
        
        if self.KSS and split_folder_name:
            split_kss_dir = MANUFACTURER_SPECTRUMS_FOLDER / split_folder_name
            print(f"[*] Initializing KSS Harmonization. Reading from: {split_kss_dir.name}")
            
            for mfr in ['GE', 'SIEMENS', 'PHILIPS']:
                kss_path = split_kss_dir / f"KSS_{mfr}.nii.gz"
                if kss_path.exists():
                    try:
                        self.kss_tensors[mfr] = torch.from_numpy(nib.load(kss_path).get_fdata(dtype=np.complex64))
                        print(f"    -> Loaded KSS for {mfr}")
                    except Exception as e:
                        print(f"    -> Error loading KSS for {mfr}: {e}")
                else:
                    print(f"    -> No KSS found for {mfr} (Expected if it's the Unseen Test domain)")
        elif self.KSS:
            print("[!] Warning: KSS requested but no 'split_folder_name' provided.")
        else:
            print("[*] KSS Harmonization disabled. Using Raw images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row['prefix']
        model_id = row['Model']
        mfr_name = self.model_to_mfr.get(model_id, 'UNKNOWN')
        gender_label = torch.tensor(row['Sex'], dtype=torch.float32)
        
        if self.KSS:
            file_path = PROJECT_ROOT / "dataset" / "kspace_amp_phase" / f"{prefix}.nii.gz"
            MRI_f = torch.from_numpy(nib.load(file_path).get_fdata(dtype=np.complex64))
            
            if mfr_name in self.kss_tensors:
                MRI_f = MRI_f - self.kss_tensors[mfr_name]
            
            MRI_f_shifted = torch.fft.ifftshift(MRI_f, dim=(-3, -2, -1))
            MRI_f = torch.fft.ifftn(MRI_f_shifted, dim=(-3, -2, -1), norm="ortho").real.float().unsqueeze(0) 
                
        else:
            sub_dir = "ADNI/ADNI_registered" if row['Project'] == 'ADNI' else "PPMI/PPMI_registered"
            file_path = PROJECT_ROOT / "dataset" / "registered" / sub_dir / f"{prefix}.nii.gz"
            MRI_f = self.transform(file_path).float() 

        return MRI_f, gender_label, torch.tensor(row['Model'], dtype=torch.long)
