"""
Model Architectures for Multimodal Breast Cancer Analysis
Includes 3D CNNs for MRI/PET, 2D CNNs for histopathology, and fusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MRI_PET_3D_CNN(nn.Module):
    """3D CNN for MRI/PET volume analysis"""
    
    def __init__(self, input_channels=2, num_classes=2, dropout_rate=0.5):
        super(MRI_PET_3D_CNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, channels, depth, height, width)
        
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class Histopathology_2D_CNN(nn.Module):
    """2D CNN for histopathology image analysis"""
    
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet50'):
        super(Histopathology_2D_CNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Modify the final layer for our number of classes
        if hasattr(self.backbone, 'fc'):
            # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            # DenseNet
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class AttentionFusion(nn.Module):
    """Attention-based fusion module for multimodal features"""
    
    def __init__(self, mri_pet_features=256, histo_features=2048, fusion_dim=512):
        super(AttentionFusion, self).__init__()
        
        self.mri_pet_features = mri_pet_features
        self.histo_features = histo_features
        self.fusion_dim = fusion_dim
        
        # Feature projection layers
        self.mri_pet_projection = nn.Linear(mri_pet_features, fusion_dim)
        self.histo_projection = nn.Linear(histo_features, fusion_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, mri_pet_feat, histo_feat):
        # Project features to common space
        mri_pet_proj = self.mri_pet_projection(mri_pet_feat)
        histo_proj = self.histo_projection(histo_feat)
        
        # Apply attention
        # Reshape for attention (batch_size, 1, features)
        mri_pet_attn = mri_pet_proj.unsqueeze(1)
        histo_attn = histo_proj.unsqueeze(1)
        
        # Concatenate for attention
        combined = torch.cat([mri_pet_attn, histo_attn], dim=1)
        
        # Apply self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Extract attended features
        mri_pet_attended = attended[:, 0, :]
        histo_attended = attended[:, 1, :]
        
        # Concatenate and fuse
        fused = torch.cat([mri_pet_attended, histo_attended], dim=1)
        fused = self.fusion_layer(fused)
        
        return fused

class MultimodalFusionModel(nn.Module):
    """Complete multimodal fusion model"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(MultimodalFusionModel, self).__init__()
        
        # Individual modality models
        self.mri_pet_model = MRI_PET_3D_CNN(input_channels=2, num_classes=num_classes, dropout_rate=dropout_rate)
        self.histo_model = Histopathology_2D_CNN(num_classes=num_classes, pretrained=True)
        
        # Fusion module
        self.fusion = AttentionFusion(
            mri_pet_features=256,  # Output from MRI_PET_3D_CNN
            histo_features=2048,   # Output from ResNet50
            fusion_dim=512
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 from fusion_dim // 2
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Modality-specific classifiers (for individual modality training)
        self.mri_pet_classifier = nn.Linear(256, num_classes)
        self.histo_classifier = nn.Linear(2048, num_classes)
    
    def forward(self, mri_pet_volume, histo_image, fusion_mode='multimodal'):
        """
        Forward pass with different fusion modes
        
        Args:
            mri_pet_volume: 3D volume (batch_size, channels, depth, height, width)
            histo_image: 2D image (batch_size, channels, height, width)
            fusion_mode: 'mri_pet_only', 'histo_only', 'multimodal'
        """
        
        if fusion_mode == 'mri_pet_only':
            return self.mri_pet_model(mri_pet_volume)
        
        elif fusion_mode == 'histo_only':
            return self.histo_model(histo_image)
        
        elif fusion_mode == 'multimodal':
            # Extract features from individual models
            mri_pet_features = self.extract_mri_pet_features(mri_pet_volume)
            histo_features = self.extract_histo_features(histo_image)
            
            # Fuse features
            fused_features = self.fusion(mri_pet_features, histo_features)
            
            # Final classification
            output = self.classifier(fused_features)
            
            return output
        
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
    
    def extract_mri_pet_features(self, x):
        """Extract features from MRI/PET model without final classification"""
        # Forward through convolutional layers
        x = F.relu(self.mri_pet_model.bn1(self.mri_pet_model.conv1(x)))
        x = self.mri_pet_model.pool(x)
        
        x = F.relu(self.mri_pet_model.bn2(self.mri_pet_model.conv2(x)))
        x = self.mri_pet_model.pool(x)
        
        x = F.relu(self.mri_pet_model.bn3(self.mri_pet_model.conv3(x)))
        x = self.mri_pet_model.pool(x)
        
        x = F.relu(self.mri_pet_model.bn4(self.mri_pet_model.conv4(x)))
        x = self.mri_pet_model.pool(x)
        
        # Global average pooling
        x = self.mri_pet_model.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x
    
    def extract_histo_features(self, x):
        """Extract features from histopathology model without final classification"""
        # Forward through backbone (remove final classification layer)
        x = self.histo_model.backbone.conv1(x)
        x = self.histo_model.backbone.bn1(x)
        x = self.histo_model.backbone.relu(x)
        x = self.histo_model.backbone.maxpool(x)
        
        x = self.histo_model.backbone.layer1(x)
        x = self.histo_model.backbone.layer2(x)
        x = self.histo_model.backbone.layer3(x)
        x = self.histo_model.backbone.layer4(x)
        
        x = self.histo_model.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class MONAI_3D_CNN(nn.Module):
    """3D CNN using MONAI framework for medical imaging"""
    
    def __init__(self, input_channels=2, num_classes=2, spatial_dims=3):
        super(MONAI_3D_CNN, self).__init__()
        
        # This is a placeholder for MONAI-based model
        # In practice, you would use MONAI's UNet, DenseNet, or other architectures
        
        # Simple 3D CNN as fallback
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create model based on type
        
        Args:
            model_type: 'mri_pet_3d', 'histo_2d', 'multimodal', 'monai_3d'
            **kwargs: Model-specific parameters
        """
        
        if model_type == 'mri_pet_3d':
            return MRI_PET_3D_CNN(**kwargs)
        
        elif model_type == 'histo_2d':
            return Histopathology_2D_CNN(**kwargs)
        
        elif model_type == 'multimodal':
            return MultimodalFusionModel(**kwargs)
        
        elif model_type == 'monai_3d':
            return MONAI_3D_CNN(**kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_model_config(model_type):
        """Get default configuration for model type"""
        
        configs = {
            'mri_pet_3d': {
                'input_channels': 2,
                'num_classes': 2,
                'dropout_rate': 0.5
            },
            'histo_2d': {
                'num_classes': 2,
                'pretrained': True,
                'model_name': 'resnet50'
            },
            'multimodal': {
                'num_classes': 2,
                'dropout_rate': 0.5
            },
            'monai_3d': {
                'input_channels': 2,
                'num_classes': 2,
                'spatial_dims': 3
            }
        }
        
        return configs.get(model_type, {})

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_models():
    """Test all model architectures"""
    logger.info("Testing model architectures...")
    
    # Test MRI/PET 3D CNN
    mri_pet_model = MRI_PET_3D_CNN()
    mri_pet_input = torch.randn(2, 2, 64, 64, 64)  # (batch, channels, depth, height, width)
    mri_pet_output = mri_pet_model(mri_pet_input)
    logger.info(f"MRI/PET 3D CNN: Input {mri_pet_input.shape} -> Output {mri_pet_output.shape}")
    logger.info(f"Parameters: {count_parameters(mri_pet_model):,}")
    
    # Test Histopathology 2D CNN
    histo_model = Histopathology_2D_CNN()
    histo_input = torch.randn(2, 3, 224, 224)  # (batch, channels, height, width)
    histo_output = histo_model(histo_input)
    logger.info(f"Histopathology 2D CNN: Input {histo_input.shape} -> Output {histo_output.shape}")
    logger.info(f"Parameters: {count_parameters(histo_model):,}")
    
    # Test Multimodal Fusion Model
    multimodal_model = MultimodalFusionModel()
    mri_pet_input = torch.randn(2, 2, 64, 64, 64)
    histo_input = torch.randn(2, 3, 224, 224)
    
    # Test different fusion modes
    mri_pet_only = multimodal_model(mri_pet_input, histo_input, 'mri_pet_only')
    histo_only = multimodal_model(mri_pet_input, histo_input, 'histo_only')
    multimodal = multimodal_model(mri_pet_input, histo_input, 'multimodal')
    
    logger.info(f"Multimodal Fusion Model:")
    logger.info(f"  MRI/PET only: {mri_pet_only.shape}")
    logger.info(f"  Histo only: {histo_only.shape}")
    logger.info(f"  Multimodal: {multimodal.shape}")
    logger.info(f"Parameters: {count_parameters(multimodal_model):,}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test all models
    test_models()
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURES TESTED SUCCESSFULLY")
    print("="*60)
    print("Available models:")
    print("- MRI_PET_3D_CNN: For 3D volume analysis")
    print("- Histopathology_2D_CNN: For 2D image analysis")
    print("- MultimodalFusionModel: For combined analysis")
    print("- MONAI_3D_CNN: MONAI-based 3D CNN")
    print("\nUse ModelFactory.create_model() to instantiate models") 