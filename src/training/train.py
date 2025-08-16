"""
Training Script for Multimodal Breast Cancer Analysis
Handles data loading, training loops, and evaluation for all model types
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_architectures import ModelFactory, MultimodalFusionModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRI_PET_Dataset(Dataset):
    """Dataset for MRI/PET 3D volumes"""
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load 3D volume
        img = sitk.ReadImage(self.file_paths[idx])
        img_array = sitk.GetArrayFromImage(img)
        
        # Convert to torch tensor (C, D, H, W)
        img_tensor = torch.from_numpy(img_array).float()
        
        # Apply transformations if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_tensor, label

class HistopathologyDataset(Dataset):
    """Dataset for histopathology 2D images"""
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.file_paths[idx]).convert('RGB')
        img_array = np.array(img)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img_array)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_tensor, label

class MultimodalDataset(Dataset):
    """Dataset for multimodal training (MRI/PET + Histopathology)"""
    
    def __init__(self, mri_pet_paths, histo_paths, labels, mri_pet_transform=None, histo_transform=None):
        self.mri_pet_paths = mri_pet_paths
        self.histo_paths = histo_paths
        self.labels = labels
        self.mri_pet_transform = mri_pet_transform
        self.histo_transform = histo_transform
    
    def __len__(self):
        return len(self.mri_pet_paths)
    
    def __getitem__(self, idx):
        # Load MRI/PET volume
        img = sitk.ReadImage(self.mri_pet_paths[idx])
        mri_pet_array = sitk.GetArrayFromImage(img)
        mri_pet_tensor = torch.from_numpy(mri_pet_array).float()
        
        if self.mri_pet_transform:
            mri_pet_tensor = self.mri_pet_transform(mri_pet_tensor)
        
        # Load histopathology image
        histo_img = Image.open(self.histo_paths[idx]).convert('RGB')
        histo_array = np.array(histo_img)
        
        if self.histo_transform:
            transformed = self.histo_transform(image=histo_array)
            histo_tensor = transformed['image']
        else:
            histo_tensor = torch.from_numpy(histo_array).permute(2, 0, 1).float() / 255.0
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return mri_pet_tensor, histo_tensor, label

class Trainer:
    """Main training class for multimodal breast cancer analysis"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Load data splits
        self.load_data_splits()
        
        # Initialize model
        self.model = self.create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def load_data_splits(self):
        """Load data splits from preprocessing"""
        splits_file = Path(self.config['data_dir']) / 'processed' / 'data_splits.json'
        
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                self.data_splits = json.load(f)
            logger.info("Loaded data splits from preprocessing")
        else:
            # Create dummy splits for testing
            logger.warning("No data splits found, creating dummy splits for testing")
            self.create_dummy_splits()
    
    def create_dummy_splits(self):
        """Create dummy data splits for testing"""
        # This is for testing when no real data is available
        self.data_splits = {
            "splits": {
                "train": {
                    "mri_pet": [],
                    "histopathology": []
                },
                "val": {
                    "mri_pet": [],
                    "histopathology": []
                },
                "test": {
                    "mri_pet": [],
                    "histopathology": []
                }
            }
        }
    
    def create_model(self):
        """Create model based on configuration"""
        model_type = self.config['model_type']
        model_config = ModelFactory.get_model_config(model_type)
        model_config.update(self.config.get('model_params', {}))
        
        model = ModelFactory.create_model(model_type, **model_config)
        logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def create_transforms(self):
        """Create data transformations"""
        if self.config['model_type'] == 'histo_2d':
            # Histopathology transforms
            train_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            return train_transform, val_transform
        
        else:
            # For 3D volumes, use simple normalization
            return None, None
    
    def create_dataloaders(self):
        """Create data loaders for training"""
        train_transform, val_transform = self.create_transforms()
        
        if self.config['model_type'] == 'multimodal':
            # Multimodal training
            train_dataset = MultimodalDataset(
                mri_pet_paths=self.data_splits['splits']['train']['mri_pet'],
                histo_paths=self.data_splits['splits']['train']['histopathology'],
                labels=[0] * len(self.data_splits['splits']['train']['mri_pet']),  # Dummy labels
                mri_pet_transform=train_transform,
                histo_transform=train_transform
            )
            
            val_dataset = MultimodalDataset(
                mri_pet_paths=self.data_splits['splits']['val']['mri_pet'],
                histo_paths=self.data_splits['splits']['val']['histopathology'],
                labels=[0] * len(self.data_splits['splits']['val']['mri_pet']),  # Dummy labels
                mri_pet_transform=val_transform,
                histo_transform=val_transform
            )
        
        elif self.config['model_type'] == 'histo_2d':
            # Histopathology training
            train_dataset = HistopathologyDataset(
                file_paths=self.data_splits['splits']['train']['histopathology'],
                labels=[0] * len(self.data_splits['splits']['train']['histopathology']),  # Dummy labels
                transform=train_transform
            )
            
            val_dataset = HistopathologyDataset(
                file_paths=self.data_splits['splits']['val']['histopathology'],
                labels=[0] * len(self.data_splits['splits']['val']['histopathology']),  # Dummy labels
                transform=val_transform
            )
        
        else:
            # MRI/PET training
            train_dataset = MRI_PET_Dataset(
                file_paths=self.data_splits['splits']['train']['mri_pet'],
                labels=[0] * len(self.data_splits['splits']['train']['mri_pet']),  # Dummy labels
                transform=train_transform
            )
            
            val_dataset = MRI_PET_Dataset(
                file_paths=self.data_splits['splits']['val']['mri_pet'],
                labels=[0] * len(self.data_splits['splits']['val']['mri_pet']),  # Dummy labels
                transform=val_transform
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, data in enumerate(progress_bar):
            if self.config['model_type'] == 'multimodal':
                mri_pet_data, histo_data, labels = data
                mri_pet_data = mri_pet_data.to(self.device)
                histo_data = histo_data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(mri_pet_data, histo_data, 'multimodal')
            
            else:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                if self.config['model_type'] == 'multimodal':
                    mri_pet_data, histo_data, labels = data
                    mri_pet_data = mri_pet_data.to(self.device)
                    histo_data = histo_data.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(mri_pet_data, histo_data, 'multimodal')
                
                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation accuracy: {self.best_val_acc:.2f}%")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        train_loader, val_loader = self.create_dataloaders()
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = self.validate_epoch(val_loader)
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Metrics/Precision', precision, epoch)
            self.writer.add_scalar('Metrics/Recall', recall, epoch)
            self.writer.add_scalar('Metrics/F1', f1, epoch)
            
            # Print epoch summary
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Final plots
        self.plot_training_curves()
        
        # Save training summary
        summary = {
            'best_val_acc': self.best_val_acc,
            'final_train_acc': self.train_accs[-1],
            'final_val_acc': self.val_accs[-1],
            'config': self.config
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multimodal breast cancer models')
    parser.add_argument('--model_type', type=str, default='multimodal', 
                       choices=['mri_pet_3d', 'histo_2d', 'multimodal', 'histopathology_2d_cnn'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Map model type names
    model_type_map = {
        'histopathology_2d_cnn': 'histo_2d',
        'mri_pet_3d_cnn': 'mri_pet_3d'
    }
    
    model_type = model_type_map.get(args.model_type, args.model_type)
    
    # Default configuration
    config = {
        'model_type': model_type,  # 'mri_pet_3d', 'histo_2d', 'multimodal'
        'data_dir': 'data',
        'output_dir': 'results/training',
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'save_frequency': 10,
        'model_params': {
            'num_classes': 2
        }
    }
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Results saved to: {trainer.output_dir}")
    print("\nNext steps:")
    print("1. Review training curves in results/training/")
    print("2. Check TensorBoard logs for detailed metrics")
    print("3. Use best_model.pth for inference")

if __name__ == "__main__":
    main() 