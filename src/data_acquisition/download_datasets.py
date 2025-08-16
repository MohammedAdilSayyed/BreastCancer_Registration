"""
Data Acquisition Script for Multimodal Breast Cancer Analysis
Downloads and organizes datasets from various sources
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.mri_pet_dir = self.base_dir / "MRI_PET"
        self.histopathology_dir = self.base_dir / "histopathology"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories
        for dir_path in [self.mri_pet_dir, self.histopathology_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, filename, chunk_size=8192):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def extract_archive(self, archive_path, extract_to):
        """Extract compressed archives"""
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.warning(f"Unknown archive format: {archive_path.suffix}")
    
    def download_tcia_qin_breast(self):
        """Download TCIA QIN-BREAST dataset (MRI + PET)"""
        logger.info("Setting up TCIA QIN-BREAST dataset download...")
        
        # Note: TCIA requires authentication and API key
        # This is a placeholder for the actual download process
        tcia_info = {
            "name": "TCIA QIN-BREAST",
            "description": "MRI and PET imaging data for breast cancer analysis",
            "url": "https://wiki.cancerimagingarchive.net/display/Public/QIN-BREAST",
            "requirements": "Requires TCIA account and API key",
            "instructions": [
                "1. Register at https://www.cancerimagingarchive.net/",
                "2. Request access to QIN-BREAST collection",
                "3. Use TCIA Data Retriever or REST API to download",
                "4. Place downloaded files in data/MRI_PET/ directory"
            ]
        }
        
        # Create instructions file
        instructions_file = self.mri_pet_dir / "TCIA_QIN_BREAST_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write(f"Dataset: {tcia_info['name']}\n")
            f.write(f"Description: {tcia_info['description']}\n")
            f.write(f"URL: {tcia_info['url']}\n")
            f.write(f"Requirements: {tcia_info['requirements']}\n\n")
            f.write("Instructions:\n")
            for instruction in tcia_info['instructions']:
                f.write(f"{instruction}\n")
        
        logger.info(f"TCIA instructions saved to {instructions_file}")
        return tcia_info
    

    
    def download_breakhis(self):
        """Download BreakHis histopathology dataset"""
        logger.info("Downloading BreakHis dataset...")
        
        # BreakHis dataset URLs (Kaggle)
        breakhis_urls = {
            "kaggle": "https://www.kaggle.com/datasets/ambarish/breakhis",
            "github": "https://github.com/ieee8023/covid-chestxray-dataset"
        }
        
        breakhis_dir = self.histopathology_dir / "BreakHis"
        breakhis_dir.mkdir(exist_ok=True)
        
        # Create instructions file
        instructions_file = breakhis_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write("Dataset: BreakHis\n")
            f.write("Description: Breast Cancer Histopathological Image Classification\n")
            f.write(f"Kaggle URL: {breakhis_urls['kaggle']}\n")
            f.write("Instructions:\n")
            f.write("1. Visit the Kaggle dataset page\n")
            f.write("2. Download the dataset (requires Kaggle account)\n")
            f.write("3. Extract files to data/histopathology/BreakHis/\n")
            f.write("4. Organize by magnification levels (40X, 100X, 200X, 400X)\n")
        
        logger.info(f"BreakHis instructions saved to {instructions_file}")
        return {"name": "BreakHis", "url": breakhis_urls['kaggle']}
    
    def download_kaggle_breast_histopathology(self):
        """Download Kaggle Breast Histopathology dataset"""
        logger.info("Setting up Kaggle Breast Histopathology dataset...")
        
        kaggle_info = {
            "name": "Kaggle Breast Histopathology",
            "description": "Breast cancer histopathology image classification dataset",
            "url": "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
            "requirements": "Requires Kaggle account and API key",
            "instructions": [
                "1. Visit the Kaggle dataset page",
                "2. Download the dataset using Kaggle API or web interface",
                "3. Extract files to data/histopathology/Kaggle_Breast_Histopathology/",
                "4. Organize by patient ID and image patches"
            ]
        }
        
        kaggle_dir = self.histopathology_dir / "Kaggle_Breast_Histopathology"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Create instructions file
        instructions_file = kaggle_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write(f"Dataset: {kaggle_info['name']}\n")
            f.write(f"Description: {kaggle_info['description']}\n")
            f.write(f"URL: {kaggle_info['url']}\n")
            f.write(f"Requirements: {kaggle_info['requirements']}\n\n")
            f.write("Instructions:\n")
            for instruction in kaggle_info['instructions']:
                f.write(f"{instruction}\n")
        
        logger.info(f"Kaggle dataset instructions saved to {instructions_file}")
        return kaggle_info
    
    def create_metadata_template(self):
        """Create metadata template for linking patient IDs across modalities"""
        logger.info("Creating metadata template...")
        
        metadata_template = {
            "patient_mapping": {
                "patient_id": "Unique patient identifier",
                "mri_scan_id": "MRI scan identifier",
                "pet_scan_id": "PET scan identifier", 
                "histopathology_slide_id": "Histopathology slide identifier",
                "diagnosis": "Cancer diagnosis (benign/malignant)",
                "tumor_grade": "Tumor grade if available",
                "tumor_stage": "Tumor stage if available"
            },
            "dataset_info": {
                "tcia_qin_breast": {
                    "modality": ["MRI", "PET"],
                    "file_format": ["NIfTI", "DICOM"],
                    "description": "Multi-modal imaging data"
                },
                "tcga_brca": {
                    "modality": ["Histopathology"],
                    "file_format": ["SVS", "TIFF"],
                    "description": "Whole-slide histopathology images"
                },
                "breakhis": {
                    "modality": ["Histopathology"],
                    "file_format": ["PNG", "JPEG"],
                    "description": "Histopathology image patches"
                },
                "kaggle_breast_histopathology": {
                    "modality": ["Histopathology"],
                    "file_format": ["PNG", "JPEG"],
                    "description": "Histopathology image patches"
                }
            }
        }
        
        # Save metadata template
        metadata_file = self.metadata_dir / "metadata_template.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata_template, f, indent=2)
        
        logger.info(f"Metadata template saved to {metadata_file}")
        return metadata_template
    
    def download_all_datasets(self):
        """Download all datasets and create organization structure"""
        logger.info("Starting dataset download process...")
        
        results = {}
        
        # Download each dataset
        results['tcia_qin_breast'] = self.download_tcia_qin_breast()
        results['breakhis'] = self.download_breakhis()
        results['kaggle_breast_histopathology'] = self.download_kaggle_breast_histopathology()
        
        # Create metadata template
        results['metadata_template'] = self.create_metadata_template()
        
        # Create summary report
        summary_file = self.base_dir / "DATASET_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write("MULTIMODAL BREAST CANCER DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, info in results.items():
                if isinstance(info, dict) and 'name' in info:
                    f.write(f"Dataset: {info['name']}\n")
                    f.write(f"URL: {info.get('url', 'N/A')}\n")
                    f.write(f"Status: Instructions created\n")
                    f.write("-" * 30 + "\n")
        
        logger.info(f"Dataset summary saved to {summary_file}")
        logger.info("Dataset download setup complete!")
        
        return results

def main():
    """Main function to run dataset download"""
    downloader = DatasetDownloader()
    results = downloader.download_all_datasets()
    
    print("\n" + "="*60)
    print("DATASET DOWNLOAD SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Follow the instructions in each dataset folder")
    print("2. Download the actual datasets using the provided URLs")
    print("3. Organize files according to the folder structure")
    print("4. Update metadata files with actual patient mappings")
    print("\nCheck the following files for detailed instructions:")
    print("- data/DATASET_SUMMARY.txt")
    print("- data/MRI_PET/TCIA_QIN_BREAST_INSTRUCTIONS.txt")
    print("- data/histopathology/*/DOWNLOAD_INSTRUCTIONS.txt")

if __name__ == "__main__":
    main() 