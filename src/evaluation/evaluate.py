"""
Evaluation Script for Multimodal Breast Cancer Analysis
Handles model evaluation, metrics calculation, and result visualization
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')

from models.model_architectures import ModelFactory
from training.train import MRI_PET_Dataset, HistopathologyDataset, MultimodalDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path, config, data_dir="data"):
        self.model_path = Path(model_path)
        self.config = config
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path("results/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Load test data
        self.test_loader = self.create_test_dataloader()
    
    def load_model(self):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with same configuration
        model_type = self.config['model_type']
        model_config = ModelFactory.get_model_config(model_type)
        model_config.update(self.config.get('model_params', {}))
        
        model = ModelFactory.create_model(model_type, **model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded successfully")
        return model
    
    def create_test_dataloader(self):
        """Create test data loader"""
        # Load data splits
        splits_file = self.data_dir / 'processed' / 'data_splits.json'
        
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                data_splits = json.load(f)
            
            test_files = data_splits['splits']['test']
        else:
            logger.warning("No data splits found, using dummy data for testing")
            test_files = {
                'mri_pet': [],
                'histopathology': []
            }
        
        # Create appropriate dataset based on model type
        if self.config['model_type'] == 'multimodal':
            dataset = MultimodalDataset(
                mri_pet_paths=test_files['mri_pet'],
                histo_paths=test_files['histopathology'],
                labels=[0] * len(test_files['mri_pet']),  # Dummy labels
                mri_pet_transform=None,
                histo_transform=None
            )
        elif self.config['model_type'] == 'histo_2d':
            dataset = HistopathologyDataset(
                file_paths=test_files['histopathology'],
                labels=[0] * len(test_files['histopathology']),  # Dummy labels
                transform=None
            )
        else:
            dataset = MRI_PET_Dataset(
                file_paths=test_files['mri_pet'],
                labels=[0] * len(test_files['mri_pet']),  # Dummy labels
                transform=None
            )
        
        test_loader = DataLoader(
            dataset, 
            batch_size=self.config.get('batch_size', 8), 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        return test_loader
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        logger.info("Starting model evaluation...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Evaluating"):
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
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Create visualizations
        self.create_evaluation_plots(all_labels, all_predictions, all_probabilities)
        
        # Save results
        self.save_evaluation_results(metrics)
        
        return metrics
    
    def calculate_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        logger.info("Calculating evaluation metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # ROC AUC (for binary classification)
        if len(np.unique(labels)) == 2:
            roc_auc = roc_auc_score(labels, [prob[1] for prob in probabilities])
        else:
            roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr')
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        class_report = classification_report(labels, predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return metrics
    
    def create_evaluation_plots(self, labels, predictions, probabilities):
        """Create comprehensive evaluation visualizations"""
        logger.info("Creating evaluation plots...")
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix(labels, predictions)
        
        # 2. ROC Curve
        if len(np.unique(labels)) == 2:
            self.plot_roc_curve(labels, probabilities)
        
        # 3. Precision-Recall Curve
        if len(np.unique(labels)) == 2:
            self.plot_precision_recall_curve(labels, probabilities)
        
        # 4. Prediction Distribution
        self.plot_prediction_distribution(labels, predictions)
        
        # 5. Metrics Summary
        self.plot_metrics_summary(labels, predictions, probabilities)
    
    def plot_confusion_matrix(self, labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels, probabilities):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, [prob[1] for prob in probabilities])
        roc_auc = roc_auc_score(labels, [prob[1] for prob in probabilities])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, labels, probabilities):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(labels, [prob[1] for prob in probabilities])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_distribution(self, labels, predictions):
        """Plot prediction distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True labels distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax1.bar(['Benign', 'Malignant'], counts, color=['lightblue', 'lightcoral'])
        ax1.set_title('True Labels Distribution')
        ax1.set_ylabel('Count')
        
        # Predicted labels distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        ax2.bar(['Benign', 'Malignant'], pred_counts, color=['lightgreen', 'lightpink'])
        ax2.set_title('Predicted Labels Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_summary(self, labels, predictions, probabilities):
        """Create comprehensive metrics summary plot"""
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        if len(np.unique(labels)) == 2:
            roc_auc = roc_auc_score(labels, [prob[1] for prob in probabilities])
        else:
            roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr')
        
        # Create subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Metrics bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [accuracy, precision, recall, f1, roc_auc]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
        
        bars = ax1.bar(metrics_names, metrics_values, color=colors)
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Confusion matrix
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        ax2.set_title('Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. ROC curve
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, [prob[1] for prob in probabilities])
            ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve')
            ax3.legend(loc="lower right")
        
        # 4. Prediction vs True labels
        ax4.scatter(labels, predictions, alpha=0.6, s=50)
        ax4.plot([0, 1], [0, 1], 'r--', lw=2)
        ax4.set_xlabel('True Labels')
        ax4.set_ylabel('Predicted Labels')
        ax4.set_title('Predicted vs True Labels')
        ax4.set_xlim(-0.1, 1.1)
        ax4.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_results(self, metrics):
        """Save evaluation results to files"""
        logger.info("Saving evaluation results...")
        
        # Save metrics as JSON
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_for_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_for_json[key] = value.tolist()
            elif isinstance(value, np.integer):
                metrics_for_json[key] = int(value)
            elif isinstance(value, np.floating):
                metrics_for_json[key] = float(value)
            else:
                metrics_for_json[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
        
        # Create summary report
        summary_file = self.output_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("MULTIMODAL BREAST CANCER MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.config['model_type']}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now()}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(metrics['labels'], metrics['predictions']))
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def create_interactive_plots(self, labels, predictions, probabilities):
        """Create interactive plots using Plotly"""
        logger.info("Creating interactive plots...")
        
        # 1. Interactive confusion matrix
        cm = confusion_matrix(labels, predictions)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        fig_cm.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        fig_cm.write_html(self.output_dir / 'confusion_matrix_interactive.html')
        
        # 2. Interactive ROC curve
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, [prob[1] for prob in probabilities])
            roc_auc = roc_auc_score(labels, [prob[1] for prob in probabilities])
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='navy', width=2, dash='dash')
            ))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05])
            )
            fig_roc.write_html(self.output_dir / 'roc_curve_interactive.html')
        
        # 3. Interactive metrics dashboard
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [accuracy, precision, recall, f1, metrics['roc_auc']]
        
        fig_dashboard = go.Figure(data=[
            go.Bar(x=metrics_names, y=metrics_values, 
                  marker_color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        ])
        fig_dashboard.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1])
        )
        fig_dashboard.write_html(self.output_dir / 'metrics_dashboard_interactive.html')
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Basic evaluation
        metrics = self.evaluate_model()
        
        # Create interactive plots
        self.create_interactive_plots(
            metrics['labels'], 
            metrics['predictions'], 
            metrics['probabilities']
        )
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION COMPLETED")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")
        
        return metrics

def main():
    """Main evaluation function"""
    
    # Example configuration (should match training config)
    config = {
        'model_type': 'multimodal',  # 'mri_pet_3d', 'histo_2d', 'multimodal'
        'data_dir': 'data',
        'batch_size': 8,
        'num_workers': 4,
        'model_params': {
            'num_classes': 2,
            'dropout_rate': 0.5
        }
    }
    
    # Model path (update this to your trained model path)
    model_path = "results/training/best_model.pth"
    
    if not Path(model_path).exists():
        logger.warning(f"Model file not found: {model_path}")
        logger.info("Please train a model first or update the model_path")
        return
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model_path, config)
    metrics = evaluator.run_comprehensive_evaluation()
    
    print("\nNext steps:")
    print("1. Review evaluation metrics in results/evaluation/")
    print("2. Check interactive plots for detailed analysis")
    print("3. Use evaluation_summary.txt for reporting")
    print("4. Compare with baseline models")

if __name__ == "__main__":
    main() 