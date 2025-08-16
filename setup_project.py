#!/usr/bin/env python3
"""
Multimodal Breast Cancer Analysis Pipeline - Setup Script
Initializes the complete project structure and provides setup guidance
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Complete project setup and initialization"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.src_dir = self.project_root / "src"
        self.results_dir = self.project_root / "results"
        self.notebooks_dir = self.project_root / "notebooks"
        
    def check_python_installation(self):
        """Check if Python and required packages are available"""
        logger.info("Checking Python installation...")
        
        try:
            import torch
            logger.info(f"✓ PyTorch {torch.__version__} is installed")
        except ImportError:
            logger.warning("✗ PyTorch is not installed")
            return False
        
        try:
            import SimpleITK
            logger.info(f"✓ SimpleITK {SimpleITK.Version()} is installed")
        except ImportError:
            logger.warning("✗ SimpleITK is not installed")
            return False
        
        try:
            import openslide
            logger.info("✓ OpenSlide is installed")
        except ImportError:
            logger.warning("✗ OpenSlide is not installed")
            return False
        
        try:
            import monai
            logger.info(f"✓ MONAI {monai.__version__} is installed")
        except ImportError:
            logger.warning("✗ MONAI is not installed")
            return False
        
        return True
    
    def create_project_structure(self):
        """Create the complete project directory structure"""
        logger.info("Creating project directory structure...")
        
        directories = [
            self.data_dir / "MRI_PET",
            self.data_dir / "histopathology",
            self.data_dir / "metadata",
            self.data_dir / "processed",
            self.src_dir / "data_acquisition",
            self.src_dir / "preprocessing",
            self.src_dir / "models",
            self.src_dir / "training",
            self.src_dir / "evaluation",
            self.results_dir / "training",
            self.results_dir / "evaluation",
            self.notebooks_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")
    
    def create_setup_instructions(self):
        """Create detailed setup instructions"""
        logger.info("Creating setup instructions...")
        
        instructions = {
            "python_setup": {
                "title": "Python Environment Setup",
                "steps": [
                    "Install Python 3.8 or higher",
                    "Create a virtual environment: python -m venv venv",
                    "Activate virtual environment:",
                    "  - Windows: venv\\Scripts\\activate",
                    "  - Linux/Mac: source venv/bin/activate",
                    "Install dependencies: pip install -r requirements.txt"
                ]
            },
            "medical_imaging_setup": {
                "title": "Medical Imaging Libraries",
                "steps": [
                    "Install SimpleITK: pip install SimpleITK",
                    "Install ANTsPy: pip install antspyx",
                    "Install OpenSlide: pip install openslide-python",
                    "Install MONAI: pip install monai",
                    "Install nibabel: pip install nibabel",
                    "Install pydicom: pip install pydicom"
                ]
            },
            "dataset_download": {
                "title": "Dataset Download Instructions",
                "steps": [
                    "Register for TCIA account: https://www.cancerimagingarchive.net/",
                    "Register for Kaggle account: https://www.kaggle.com/",
                    "Follow instructions in data/ folders for each dataset",
                    "Download TCIA QIN-BREAST dataset",
                    "Download TCGA-BRCA histopathology slides",
                    "Download BreakHis dataset from Kaggle",
                    "Download Kaggle breast histopathology dataset"
                ]
            },
            "usage_workflow": {
                "title": "Usage Workflow",
                "steps": [
                    "1. Download datasets: python src/data_acquisition/download_datasets.py",
                    "2. Explore data: python src/data_acquisition/data_exploration.py",
                    "3. Preprocess data: python src/preprocessing/preprocess.py",
                    "4. Train models: python src/training/train.py",
                    "5. Evaluate results: python src/evaluation/evaluate.py"
                ]
            }
        }
        
        # Save instructions
        instructions_file = self.project_root / "SETUP_INSTRUCTIONS.json"
        with open(instructions_file, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        logger.info(f"Setup instructions saved to: {instructions_file}")
        return instructions
    
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        guide_content = """# Quick Start Guide

## 1. Environment Setup
```bash
# Install Python 3.8+
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Datasets
Follow the instructions in each dataset folder:
- `data/MRI_PET/TCIA_QIN_BREAST_INSTRUCTIONS.txt`
- `data/histopathology/TCGA_BRCA/DOWNLOAD_INSTRUCTIONS.txt`
- `data/histopathology/BreakHis/DOWNLOAD_INSTRUCTIONS.txt`
- `data/histopathology/Kaggle_Breast_Histopathology/DOWNLOAD_INSTRUCTIONS.txt`

## 3. Run the Pipeline
```bash
# Explore data
python src/data_acquisition/data_exploration.py

# Preprocess data
python src/preprocessing/preprocess.py

# Train models
python src/training/train.py

# Evaluate results
python src/evaluation/evaluate.py
```

## 4. View Results
- Check `results/training/` for training logs and models
- Check `results/evaluation/` for evaluation results
- Use TensorBoard: `tensorboard --logdir results/training/tensorboard`

## 5. Explore with Jupyter
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## Troubleshooting
- Ensure all medical imaging libraries are installed
- Check dataset download instructions carefully
- Verify Python environment and dependencies
- Review logs in results/ directory for errors
"""
        
        guide_file = self.project_root / "QUICK_START.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Quick start guide saved to: {guide_file}")
    
    def run_initial_tests(self):
        """Run initial tests to verify setup"""
        logger.info("Running initial tests...")
        
        tests = []
        
        # Test 1: Check if all directories exist
        required_dirs = [
            self.data_dir, self.src_dir, self.results_dir, self.notebooks_dir
        ]
        for directory in required_dirs:
            if directory.exists():
                tests.append(f"✓ Directory exists: {directory}")
            else:
                tests.append(f"✗ Directory missing: {directory}")
        
        # Test 2: Check if source files exist
        required_files = [
            self.src_dir / "data_acquisition" / "download_datasets.py",
            self.src_dir / "preprocessing" / "preprocess.py",
            self.src_dir / "models" / "model_architectures.py",
            self.src_dir / "training" / "train.py",
            self.src_dir / "evaluation" / "evaluate.py"
        ]
        
        for file_path in required_files:
            if file_path.exists():
                tests.append(f"✓ File exists: {file_path}")
            else:
                tests.append(f"✗ File missing: {file_path}")
        
        # Test 3: Check requirements.txt
        if (self.project_root / "requirements.txt").exists():
            tests.append("✓ requirements.txt exists")
        else:
            tests.append("✗ requirements.txt missing")
        
        return tests
    
    def print_setup_summary(self):
        """Print a comprehensive setup summary"""
        print("\n" + "="*80)
        print("MULTIMODAL BREAST CANCER ANALYSIS PIPELINE - SETUP COMPLETE")
        print("="*80)
        
        print("\n📁 Project Structure Created:")
        print(f"  • Data directory: {self.data_dir}")
        print(f"  • Source code: {self.src_dir}")
        print(f"  • Results: {self.results_dir}")
        print(f"  • Notebooks: {self.notebooks_dir}")
        
        print("\n📋 Next Steps:")
        print("  1. Install Python dependencies: pip install -r requirements.txt")
        print("  2. Download datasets (follow instructions in data/ folders)")
        print("  3. Run data exploration: python src/data_acquisition/data_exploration.py")
        print("  4. Start preprocessing: python src/preprocessing/preprocess.py")
        print("  5. Train models: python src/training/train.py")
        print("  6. Evaluate results: python src/evaluation/evaluate.py")
        
        print("\n📚 Documentation:")
        print("  • README.md - Main project documentation")
        print("  • PROJECT_SUMMARY.md - Comprehensive project overview")
        print("  • QUICK_START.md - Quick start guide")
        print("  • SETUP_INSTRUCTIONS.json - Detailed setup instructions")
        
        print("\n🔧 Key Features:")
        print("  • Multi-modal image registration (MRI + PET)")
        print("  • Histopathology slide processing and tiling")
        print("  • 3D CNN for volumetric data analysis")
        print("  • 2D CNN with transfer learning for histopathology")
        print("  • Attention-based multimodal fusion")
        print("  • Comprehensive evaluation and visualization")
        
        print("\n📊 Expected Results:")
        print("  • Target accuracy: >85% for multimodal fusion")
        print("  • Target ROC-AUC: >0.90")
        print("  • Comprehensive evaluation metrics")
        print("  • Interactive visualizations and reports")
        
        print("\n🎯 Research Impact:")
        print("  • Improved breast cancer diagnosis accuracy")
        print("  • Reduced false positives and negatives")
        print("  • Better patient stratification")
        print("  • Interpretable results for clinicians")
        
        print("\n" + "="*80)
        print("Setup completed successfully! 🎉")
        print("="*80)
    
    def setup_complete_project(self):
        """Complete project setup"""
        logger.info("Starting complete project setup...")
        
        # Create project structure
        self.create_project_structure()
        
        # Create setup instructions
        self.create_setup_instructions()
        
        # Create quick start guide
        self.create_quick_start_guide()
        
        # Run initial tests
        tests = self.run_initial_tests()
        
        # Print test results
        print("\nSetup Tests:")
        for test in tests:
            print(f"  {test}")
        
        # Print summary
        self.print_setup_summary()

def main():
    """Main setup function"""
    print("Multimodal Breast Cancer Analysis Pipeline - Setup")
    print("="*60)
    
    setup = ProjectSetup()
    
    # Check Python installation
    if not setup.check_python_installation():
        print("\n⚠️  Warning: Some required packages are not installed.")
        print("Please install them using: pip install -r requirements.txt")
        print("Continuing with setup...\n")
    
    # Complete setup
    setup.setup_complete_project()
    
    print("\n✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download datasets (follow instructions in data/ folders)")
    print("3. Start with: python src/data_acquisition/data_exploration.py")

if __name__ == "__main__":
    main() 