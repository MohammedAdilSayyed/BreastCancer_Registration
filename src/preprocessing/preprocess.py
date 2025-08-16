"""
Preprocessing Pipeline for Multimodal Breast Cancer Analysis
Handles image registration, normalization, and data preparation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import nibabel as nib
from scipy import ndimage
import cv2
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalPreprocessor:
    def __init__(self, data_dir="src/data_acquisition/data", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.mri_pet_dir = self.data_dir / "MRI_PET"
        self.histopathology_dir = self.data_dir / "histopathology"
        
        # Create output directories
        for subdir in ["mri_pet", "histopathology", "registered", "metadata"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def preprocess_mri_pet(self, input_file, output_file, modality="MRI"):
        """Preprocess MRI/PET images"""
        logger.info(f"Preprocessing {modality} image: {input_file}")
        
        try:
            # Load image
            if input_file.suffix.lower() in ['.nii', '.nii.gz']:
                img = sitk.ReadImage(str(input_file))
            elif input_file.suffix.lower() in ['.dcm', '.dicom']:
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(input_file))
                img = reader.Execute()
            else:
                logger.warning(f"Unsupported file format: {input_file}")
                return False
            
            # Get image array
            img_array = sitk.GetArrayFromImage(img)
            
            # Basic preprocessing
            # 1. Normalize intensity
            img_array = self.normalize_intensity(img_array, modality)
            
            # 2. Resample to standard resolution if needed
            img_array = self.resample_volume(img_array, target_spacing=(1.0, 1.0, 1.0))
            
            # 3. Apply noise reduction
            img_array = self.reduce_noise(img_array)
            
            # 4. Save processed image
            processed_img = sitk.GetImageFromArray(img_array)
            processed_img.CopyInformation(img)  # Preserve metadata
            
            sitk.WriteImage(processed_img, str(output_file))
            
            logger.info(f"Successfully preprocessed {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing {input_file}: {e}")
            return False
    
    def normalize_intensity(self, img_array, modality):
        """Normalize image intensity based on modality"""
        if modality == "MRI":
            # Z-score normalization for MRI
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)
            if std_val > 0:
                img_array = (img_array - mean_val) / std_val
        elif modality == "PET":
            # SUV normalization for PET
            max_val = np.max(img_array)
            if max_val > 0:
                img_array = img_array / max_val
        
        return img_array
    
    def resample_volume(self, img_array, target_spacing=(1.0, 1.0, 1.0)):
        """Resample volume to target spacing"""
        # Simple resampling - in practice, use proper interpolation
        current_shape = img_array.shape
        target_shape = tuple(int(s * 1.0) for s in current_shape)  # Simplified
        
        if current_shape != target_shape:
            img_array = ndimage.zoom(img_array, 
                                   [t/c for t, c in zip(target_shape, current_shape)],
                                   order=1)
        
        return img_array
    
    def reduce_noise(self, img_array):
        """Apply noise reduction"""
        # Gaussian smoothing
        img_array = ndimage.gaussian_filter(img_array, sigma=0.5)
        return img_array
    
    def register_mri_pet(self, mri_file, pet_file, output_dir):
        """Register MRI and PET images using ANTsPy"""
        logger.info(f"Registering MRI: {mri_file} and PET: {pet_file}")
        
        try:
            # Load images
            mri_img = sitk.ReadImage(str(mri_file))
            pet_img = sitk.ReadImage(str(pet_file))
            
            # Simple registration using SimpleITK
            # In practice, use ANTsPy for more sophisticated registration
            
            # 1. Resample PET to MRI space
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetReferenceImage(mri_img)
            resample_filter.SetInterpolator(sitk.sitkLinear)
            pet_registered = resample_filter.Execute(pet_img)
            
            # 2. Save registered images
            mri_output = output_dir / f"{mri_file.stem}_registered.nii.gz"
            pet_output = output_dir / f"{pet_file.stem}_registered.nii.gz"
            
            sitk.WriteImage(mri_img, str(mri_output))
            sitk.WriteImage(pet_registered, str(pet_output))
            
            # 3. Create transformation matrix info
            transform_info = {
                "mri_file": str(mri_file),
                "pet_file": str(pet_file),
                "mri_registered": str(mri_output),
                "pet_registered": str(pet_output),
                "registration_method": "SimpleITK Resample",
                "registration_date": pd.Timestamp.now().isoformat()
            }
            
            transform_file = output_dir / f"registration_info_{mri_file.stem}.json"
            with open(transform_file, 'w') as f:
                json.dump(transform_info, f, indent=2)
            
            logger.info(f"Registration completed: {mri_output}, {pet_output}")
            return True
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def preprocess_histopathology(self, input_file, output_dir, patch_size=512):
        """Preprocess histopathology images by tiling into patches"""
        logger.info(f"Preprocessing histopathology: {input_file}")
        
        try:
            # Load whole-slide image
            slide = openslide.OpenSlide(str(input_file))
            
            # Get slide properties
            slide_width, slide_height = slide.dimensions
            level_count = slide.level_count
            
            # Create output directory for this slide
            slide_output_dir = output_dir / input_file.stem
            slide_output_dir.mkdir(exist_ok=True)
            
            # Tile the slide into patches
            patches = []
            patch_info = []
            
            # Use level 1 (downsampled) for faster processing
            level = min(1, level_count - 1)
            level_width, level_height = slide.level_dimensions[level]
            
            # Calculate number of patches
            num_patches_x = level_width // patch_size
            num_patches_y = level_height // patch_size
            
            logger.info(f"Creating {num_patches_x * num_patches_y} patches of size {patch_size}x{patch_size}")
            
            for i in tqdm(range(num_patches_x), desc="Processing patches"):
                for j in range(num_patches_y):
                    # Extract patch
                    x = i * patch_size
                    y = j * patch_size
                    
                    patch = slide.read_region((x, y), level, (patch_size, patch_size))
                    patch = patch.convert('RGB')
                    
                    # Basic preprocessing
                    patch_array = np.array(patch)
                    
                    # Color normalization
                    patch_array = self.normalize_histopathology_color(patch_array)
                    
                    # Save patch
                    patch_filename = f"patch_{i:03d}_{j:03d}.png"
                    patch_path = slide_output_dir / patch_filename
                    
                    patch_img = Image.fromarray(patch_array.astype(np.uint8))
                    patch_img.save(patch_path)
                    
                    # Store patch information
                    patch_info.append({
                        "patch_id": f"{i:03d}_{j:03d}",
                        "position": (x, y),
                        "level": level,
                        "size": (patch_size, patch_size),
                        "file_path": str(patch_path)
                    })
            
            # Save patch metadata
            metadata_file = slide_output_dir / "patch_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    "slide_file": str(input_file),
                    "slide_dimensions": (slide_width, slide_height),
                    "level_count": level_count,
                    "patch_size": patch_size,
                    "num_patches": len(patch_info),
                    "patches": patch_info
                }, f, indent=2)
            
            slide.close()
            logger.info(f"Successfully processed {len(patch_info)} patches")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing histopathology {input_file}: {e}")
            return False
    
    def normalize_histopathology_color(self, img_array):
        """Normalize histopathology image color"""
        # Convert to LAB color space for better color normalization
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Normalize each channel
        for i in range(3):
            lab[:, :, i] = cv2.normalize(lab[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to RGB
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return normalized
    
    def preprocess_histopathology_image(self, input_file, output_dir):
        """Preprocess regular histopathology image files (PNG, JPG, etc.)"""
        logger.info(f"Preprocessing histopathology image: {input_file}")
        
        try:
            # Load image using PIL
            img = Image.open(str(input_file))
            img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Basic preprocessing
            # 1. Color normalization
            img_array = self.normalize_histopathology_color(img_array)
            
            # 2. Resize to standard size if needed (optional)
            target_size = (512, 512)
            if img_array.shape[:2] != target_size:
                img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # 3. Apply noise reduction
            img_array = cv2.medianBlur(img_array, 3)
            
            # 4. Save processed image
            output_file = output_dir / f"{input_file.stem}_processed.png"
            processed_img = Image.fromarray(img_array.astype(np.uint8))
            processed_img.save(output_file)
            
            # 5. Create metadata
            metadata = {
                "original_file": str(input_file),
                "processed_file": str(output_file),
                "original_size": img.size,
                "processed_size": target_size,
                "preprocessing_steps": ["color_normalization", "resize", "noise_reduction"],
                "processing_date": pd.Timestamp.now().isoformat()
            }
            
            metadata_file = output_dir / f"{input_file.stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing histopathology image {input_file}: {e}")
            return False
    
    def create_data_splits(self, processed_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
        """Create train/validation/test splits"""
        logger.info("Creating data splits...")
        
        # Collect all processed files
        mri_pet_files = list((processed_dir / "mri_pet").rglob("*.nii.gz"))
        histo_files = list((processed_dir / "histopathology").rglob("*.png"))
        
        # Create splits
        splits = {
            "train": {"mri_pet": [], "histopathology": []},
            "val": {"mri_pet": [], "histopathology": []},
            "test": {"mri_pet": [], "histopathology": []}
        }
        
        # Split MRI/PET files
        np.random.shuffle(mri_pet_files)
        n_mri_pet = len(mri_pet_files)
        n_train = int(n_mri_pet * train_ratio)
        n_val = int(n_mri_pet * val_ratio)
        
        splits["train"]["mri_pet"] = mri_pet_files[:n_train]
        splits["val"]["mri_pet"] = mri_pet_files[n_train:n_train + n_val]
        splits["test"]["mri_pet"] = mri_pet_files[n_train + n_val:]
        
        # Split histopathology files
        np.random.shuffle(histo_files)
        n_histo = len(histo_files)
        n_train = int(n_histo * train_ratio)
        n_val = int(n_histo * val_ratio)
        
        splits["train"]["histopathology"] = histo_files[:n_train]
        splits["val"]["histopathology"] = histo_files[n_train:n_train + n_val]
        splits["test"]["histopathology"] = histo_files[n_train + n_val:]
        
        # Save split information
        split_file = output_dir / "data_splits.json"
        with open(split_file, 'w') as f:
            json.dump({
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": 1 - train_ratio - val_ratio,
                "splits": {
                    split: {
                        modality: [str(f) for f in files]
                        for modality, files in data.items()
                    }
                    for split, data in splits.items()
                }
            }, f, indent=2)
        
        logger.info(f"Data splits saved to {split_file}")
        return splits
    
    def create_augmentation_pipeline(self):
        """Create data augmentation pipeline"""
        # For histopathology images
        histo_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.1),
            ToTensorV2()
        ])
        
        # For MRI/PET volumes (3D augmentation)
        # Note: This is a simplified version. In practice, use MONAI for 3D augmentation
        mri_pet_augmentation = {
            "flip": lambda x: np.flip(x, axis=np.random.randint(0, 3)),
            "rotation": lambda x: ndimage.rotate(x, angle=np.random.uniform(-15, 15), axes=(1, 2)),
            "noise": lambda x: x + np.random.normal(0, 0.01, x.shape)
        }
        
        return histo_augmentation, mri_pet_augmentation
    
    def preprocess_all_data(self):
        """Main preprocessing pipeline"""
        logger.info("Starting comprehensive preprocessing pipeline...")
        
        # Skip MRI/PET processing since already done
        mri_pet_processed = 0
        logger.info("Skipping MRI/PET processing - already completed")
        
        # Process histopathology data
        histo_processed = 0
        if self.histopathology_dir.exists():
            for file_path in self.histopathology_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.svs', '.tiff', '.tif', '.png', '.jpg', '.jpeg']:
                    if self.preprocess_histopathology_image(file_path, self.output_dir / "histopathology"):
                        histo_processed += 1
        
        # Create data splits
        splits = self.create_data_splits(self.output_dir, self.output_dir)
        
        # Create augmentation pipeline
        histo_aug, mri_pet_aug = self.create_augmentation_pipeline()
        
        # Save preprocessing summary
        summary = {
            "preprocessing_date": pd.Timestamp.now().isoformat(),
            "mri_pet_processed": mri_pet_processed,
            "histopathology_processed": histo_processed,
            "output_directory": str(self.output_dir),
            "augmentation_pipelines_created": True
        }
        
        summary_file = self.output_dir / "preprocessing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Preprocessing pipeline completed!")
        logger.info(f"Processed {mri_pet_processed} MRI/PET files")
        logger.info(f"Processed {histo_processed} histopathology files")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return summary

def main():
    """Main function to run preprocessing"""
    preprocessor = MultimodalPreprocessor()
    summary = preprocessor.preprocess_all_data()
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED")
    print("="*60)
    print(f"MRI/PET files processed: {summary['mri_pet_processed']}")
    print(f"Histopathology files processed: {summary['histopathology_processed']}")
    print(f"Output directory: {summary['output_directory']}")
    print("\nNext steps:")
    print("1. Review preprocessing_summary.json for details")
    print("2. Check data_splits.json for train/val/test splits")
    print("3. Proceed to model development")

if __name__ == "__main__":
    main() 