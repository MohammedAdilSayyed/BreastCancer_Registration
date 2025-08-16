"""
Data Exploration Script for Multimodal Breast Cancer Analysis
Analyzes downloaded datasets and extracts metadata
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import openslide
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import nibabel as nib
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExplorer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.mri_pet_dir = self.data_dir / "MRI_PET"
        self.histopathology_dir = self.data_dir / "histopathology"
        self.metadata_dir = self.data_dir / "metadata"
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def explore_mri_pet_data(self):
        """Explore MRI and PET imaging data"""
        logger.info("Exploring MRI and PET data...")
        
        mri_pet_info = {
            "total_files": 0,
            "file_types": defaultdict(int),
            "file_sizes": [],
            "modalities": defaultdict(int),
            "sample_data": {}
        }
        
        if not self.mri_pet_dir.exists():
            logger.warning(f"MRI_PET directory not found: {self.mri_pet_dir}")
            return mri_pet_info
        
        # Walk through MRI_PET directory
        for file_path in self.mri_pet_dir.rglob("*"):
            if file_path.is_file():
                mri_pet_info["total_files"] += 1
                file_size = file_path.stat().st_size
                mri_pet_info["file_sizes"].append(file_size)
                
                # Categorize by file extension
                ext = file_path.suffix.lower()
                mri_pet_info["file_types"][ext] += 1
                
                # Try to identify modality from filename
                filename = file_path.name.lower()
                if "mri" in filename or "t1" in filename or "t2" in filename:
                    mri_pet_info["modalities"]["MRI"] += 1
                elif "pet" in filename or "fdg" in filename:
                    mri_pet_info["modalities"]["PET"] += 1
                
                # Sample a few files for detailed analysis
                if len(mri_pet_info["sample_data"]) < 5:
                    try:
                        if ext in ['.nii', '.nii.gz']:
                            # NIfTI file
                            img = nib.load(str(file_path))
                            mri_pet_info["sample_data"][file_path.name] = {
                                "type": "NIfTI",
                                "shape": img.shape,
                                "dtype": str(img.get_data_dtype()),
                                "affine": img.affine.tolist() if img.affine is not None else None
                            }
                        elif ext in ['.dcm', '.dicom']:
                            # DICOM file
                            mri_pet_info["sample_data"][file_path.name] = {
                                "type": "DICOM",
                                "size": file_size
                            }
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        return mri_pet_info
    
    def explore_histopathology_data(self):
        """Explore histopathology image data"""
        logger.info("Exploring histopathology data...")
        
        histo_info = {
            "total_files": 0,
            "file_types": defaultdict(int),
            "file_sizes": [],
            "datasets": defaultdict(lambda: {"count": 0, "sizes": []}),
            "sample_data": {}
        }
        
        if not self.histopathology_dir.exists():
            logger.warning(f"Histopathology directory not found: {self.histopathology_dir}")
            return histo_info
        
        # Walk through histopathology directory
        for file_path in self.histopathology_dir.rglob("*"):
            if file_path.is_file():
                histo_info["total_files"] += 1
                file_size = file_path.stat().st_size
                histo_info["file_sizes"].append(file_size)
                
                # Categorize by file extension
                ext = file_path.suffix.lower()
                histo_info["file_types"][ext] += 1
                
                # Identify dataset from path
                for dataset in ["BreakHis", "Kaggle_Breast_Histopathology"]:
                    if dataset.lower() in str(file_path).lower():
                        histo_info["datasets"][dataset]["count"] += 1
                        histo_info["datasets"][dataset]["sizes"].append(file_size)
                        break
                
                # Sample a few files for detailed analysis
                if len(histo_info["sample_data"]) < 10:
                    try:
                        if ext in ['.svs', '.tiff', '.tif']:
                            # Whole-slide image
                            slide = openslide.OpenSlide(str(file_path))
                            histo_info["sample_data"][file_path.name] = {
                                "type": "Whole-slide",
                                "dimensions": slide.dimensions,
                                "level_count": slide.level_count,
                                "level_dimensions": slide.level_dimensions,
                                "properties": dict(slide.properties)
                            }
                            slide.close()
                        elif ext in ['.png', '.jpg', '.jpeg']:
                            # Regular image
                            img = Image.open(file_path)
                            histo_info["sample_data"][file_path.name] = {
                                "type": "Image",
                                "size": img.size,
                                "mode": img.mode,
                                "format": img.format
                            }
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        return histo_info
    
    def create_data_summary(self, mri_pet_info, histo_info):
        """Create comprehensive data summary"""
        logger.info("Creating data summary...")
        
        summary = {
            "exploration_date": pd.Timestamp.now().isoformat(),
            "mri_pet_data": mri_pet_info,
            "histopathology_data": histo_info,
            "statistics": {
                "total_files": mri_pet_info["total_files"] + histo_info["total_files"],
                "total_size_mb": sum(mri_pet_info["file_sizes"] + histo_info["file_sizes"]) / (1024 * 1024)
            }
        }
        
        # Save summary
        summary_file = self.results_dir / "data_exploration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Data summary saved to {summary_file}")
        return summary
    
    def create_visualizations(self, mri_pet_info, histo_info):
        """Create data visualization plots"""
        logger.info("Creating data visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multimodal Breast Cancer Dataset Exploration', fontsize=16)
        
        # 1. File type distribution for MRI/PET
        if mri_pet_info["file_types"]:
            axes[0, 0].pie(mri_pet_info["file_types"].values(), 
                          labels=mri_pet_info["file_types"].keys(), 
                          autopct='%1.1f%%')
            axes[0, 0].set_title('MRI/PET File Types')
        
        # 2. File type distribution for histopathology
        if histo_info["file_types"]:
            axes[0, 1].pie(histo_info["file_types"].values(), 
                          labels=histo_info["file_types"].keys(), 
                          autopct='%1.1f%%')
            axes[0, 1].set_title('Histopathology File Types')
        
        # 3. Dataset distribution
        if histo_info["datasets"]:
            dataset_names = list(histo_info["datasets"].keys())
            dataset_counts = [histo_info["datasets"][name]["count"] for name in dataset_names]
            axes[1, 0].bar(dataset_names, dataset_counts)
            axes[1, 0].set_title('Histopathology Dataset Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. File size distribution
        all_sizes = mri_pet_info["file_sizes"] + histo_info["file_sizes"]
        if all_sizes:
            axes[1, 1].hist(all_sizes, bins=50, alpha=0.7)
            axes[1, 1].set_title('File Size Distribution')
            axes[1, 1].set_xlabel('File Size (bytes)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "data_exploration_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {plot_file}")
    
    def generate_metadata_report(self, summary):
        """Generate metadata report for patient linking"""
        logger.info("Generating metadata report...")
        
        report = {
            "dataset_overview": {
                "mri_pet_files": summary["mri_pet_data"]["total_files"],
                "histopathology_files": summary["histopathology_data"]["total_files"],
                "total_size_gb": summary["statistics"]["total_size_mb"] / 1024
            },
            "file_formats": {
                "mri_pet": dict(summary["mri_pet_data"]["file_types"]),
                "histopathology": dict(summary["histopathology_data"]["file_types"])
            },
            "sample_data_preview": {
                "mri_pet_samples": len(summary["mri_pet_data"]["sample_data"]),
                "histopathology_samples": len(summary["histopathology_data"]["sample_data"])
            },
            "recommendations": [
                "Create patient ID mapping between MRI/PET and histopathology data",
                "Standardize file naming conventions across datasets",
                "Implement data validation for file integrity",
                "Set up preprocessing pipeline for each modality"
            ]
        }
        
        # Save report
        report_file = self.results_dir / "metadata_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Metadata report saved to {report_file}")
        return report
    
    def explore_all_data(self):
        """Main function to explore all datasets"""
        logger.info("Starting comprehensive data exploration...")
        
        # Explore each modality
        mri_pet_info = self.explore_mri_pet_data()
        histo_info = self.explore_histopathology_data()
        
        # Create summary
        summary = self.create_data_summary(mri_pet_info, histo_info)
        
        # Create visualizations
        self.create_visualizations(mri_pet_info, histo_info)
        
        # Generate metadata report
        report = self.generate_metadata_report(summary)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)
        print(f"Total files found: {summary['statistics']['total_files']}")
        print(f"Total size: {summary['statistics']['total_size_mb']:.2f} MB")
        print(f"MRI/PET files: {mri_pet_info['total_files']}")
        print(f"Histopathology files: {histo_info['total_files']}")
        
        if mri_pet_info["file_types"]:
            print(f"\nMRI/PET file types: {dict(mri_pet_info['file_types'])}")
        
        if histo_info["file_types"]:
            print(f"Histopathology file types: {dict(histo_info['file_types'])}")
        
        if histo_info["datasets"]:
            print(f"\nHistopathology datasets:")
            for dataset, info in histo_info["datasets"].items():
                print(f"  {dataset}: {info['count']} files")
        
        print(f"\nResults saved to: {self.results_dir}")
        
        return summary, report

def main():
    """Main function to run data exploration"""
    explorer = DataExplorer()
    summary, report = explorer.explore_all_data()
    
    print("\nNext steps:")
    print("1. Review the generated reports in results/ directory")
    print("2. Check data_exploration_summary.json for detailed file analysis")
    print("3. Examine metadata_report.json for patient linking recommendations")
    print("4. View data_exploration_plots.png for visual insights")

if __name__ == "__main__":
    main() 