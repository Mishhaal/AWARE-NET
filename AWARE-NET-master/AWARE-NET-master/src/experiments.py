import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from visualization import Visualizer
from tqdm import tqdm

class ExperimentRunner:
    def __init__(self, config, model, trainer):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.results_dir = config.RESULTS_DIR
        self.logger = logging.getLogger(__name__)
        
    def run_experiments(self, test_loader):
        try:
            # Train the model first
            self.trainer.train()
            
            # Parse variant name safely and consistently
            variant_parts = self.trainer.variant_name.split('_')
            dataset = variant_parts[0]  # ff++ or celebdf
            model_type = variant_parts[1]  # xception, res2net, etc.
            is_no_aug = 'no_augmentation' in self.trainer.variant_name
            aug_type = 'no_aug' if is_no_aug else 'with_aug'
            
            self.logger.info(f"\nRunning experiment for:")
            self.logger.info(f"Dataset: {dataset}")
            self.logger.info(f"Model: {model_type}")
            self.logger.info(f"Augmentation: {'disabled' if is_no_aug else 'enabled'}")
            
            # Test on the dataset
            metrics = self._evaluate_dataset(test_loader, self.trainer.variant_name)
            
            # Save results in the correct directory
            results_dir = self.results_dir / 'metrics' / dataset / model_type / aug_type
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Double check we're saving in the right place
            self.logger.info(f"Saving results to: {results_dir}")
            self.logger.info(f"Current run is {'without' if is_no_aug else 'with'} augmentation")
            
            # Save results
            self._save_results(metrics, results_dir)
            
            # Generate visualizations
            plots_dir = self.results_dir / 'plots' / dataset / model_type / aug_type
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Pass augmentation information to visualization
            self._generate_visualizations(
                metrics=metrics, 
                plots_dir=plots_dir,
                is_no_aug=is_no_aug,
                dataset=dataset,
                model_type=model_type
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in run_experiments: {str(e)}")
            raise
    
    def _evaluate_dataset(self, loader, variant_name):
        self.logger.info(f"Starting evaluation for {variant_name}")
        self.model.eval()
        predictions = []
        true_labels = []
        
        # Add progress bar for evaluation
        eval_pbar = tqdm(loader, desc='Evaluating', leave=True)
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images = images.to(self.trainer.device)
                outputs = self.model(images)
                
                # Handle both single model and ensemble outputs
                if isinstance(outputs, torch.Tensor) and outputs.shape[1] == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                else:
                    probs = outputs.squeeze()
                
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.numpy())
                
                # Update progress bar with current batch size
                eval_pbar.set_postfix({
                    'batch_size': images.size(0),
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                })
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        try:
            # Calculate metrics with progress updates
            self.logger.info("Calculating metrics...")
            
            auc = roc_auc_score(true_labels, predictions)
            pred_labels = (predictions > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
            accuracy = accuracy_score(true_labels, pred_labels)
            
            metrics = {
                'performance_metrics': {
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                },
                'raw_data': {
                    'predictions': predictions.tolist(),
                    'true_labels': true_labels.tolist()
                },
                'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
            }
            
            # Log metrics with clear formatting
            self.logger.info(f"\nEvaluation metrics for {variant_name}:")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"AUC:       {auc:.4f}")
            self.logger.info(f"Accuracy:  {accuracy:.4f}")
            self.logger.info(f"F1 Score:  {f1:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall:    {recall:.4f}")
            self.logger.info(f"{'='*50}")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            self.logger.error(f"Predictions shape: {predictions.shape}")
            self.logger.error(f"Labels shape: {true_labels.shape}")
            raise
        
        return metrics
    
    def _save_results(self, metrics, results_dir):
        # Save test metrics
        test_results_path = results_dir / 'test_results.json'
        self.logger.info(f"Saving test results to {test_results_path}")
        
        # Add timestamp and configuration to results
        full_results = {
            'test_metrics': metrics,
            'configuration': {
                'model_type': type(self.model).__name__,
                'dataset_fraction': self.config.DATASET_FRACTION,
                'image_size': self.config.IMAGE_SIZE,
                'batch_size': self.config.BATCH_SIZE
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(test_results_path, 'w') as f:
            json.dump(full_results, f, indent=4)
        
        # Save model weights if it's an ensemble
        if hasattr(self.model, 'get_model_weights'):
            weights = self.model.get_model_weights()
            if weights is not None:
                weights_path = results_dir / 'ensemble_weights.json'
                with open(weights_path, 'w') as f:
                    json.dump({
                        'xception': float(weights[0]),
                        'res2net': float(weights[1]),
                        'efficientnet': float(weights[2])
                    }, f, indent=4)
                self.logger.info(f"Saved ensemble weights to {weights_path}")
    
    def _generate_visualizations(self, metrics, plots_dir, is_no_aug, dataset, model_type):
        self.logger.info(f"Generating visualizations in {plots_dir}")
        
        # Create Visualizer instance
        visualizer = Visualizer(self.config)
        
        # Get training history from trainer
        metrics_history = self.trainer.metrics_history
        
        # Verify metrics data
        self.logger.info("\nVerifying metrics data before visualization:")
        self.logger.info("Training History:")
        for key, values in metrics_history.items():
            self.logger.info(f"{key}: {len(values)} values, range: [{min(values):.4f}, {max(values):.4f}]")
        
        self.logger.info("\nTest Metrics:")
        self.logger.info(f"AUC: {metrics['performance_metrics']['auc']:.4f}")
        self.logger.info(f"Accuracy: {metrics['performance_metrics']['accuracy']:.4f}")
        
        try:
            # Generate all plots with consistent augmentation info
            visualizer.generate_all_plots(
                metrics_history=metrics_history,
                test_results={
                    'performance_metrics': metrics['performance_metrics'],
                    'raw_data': {
                        'predictions': metrics['raw_data']['predictions'],
                        'true_labels': metrics['raw_data']['true_labels']
                    },
                    'confusion_matrix': metrics['confusion_matrix']
                },
                model_weights=self.model.get_model_weights() if hasattr(self.model, 'get_model_weights') else None,
                dataset_name=dataset,
                model_type=model_type,
                is_no_aug=is_no_aug
            )
            
            self.logger.info("Successfully generated all plots")
            
        except Exception as e:
            self.logger.error(f"Error in visualization generation: {str(e)}")
            self.logger.error("Metrics History:")
            for key, value in metrics_history.items():
                if isinstance(value, list):
                    self.logger.error(f"{key}: {len(value)} entries, sample: {value[:3]}")
                else:
                    self.logger.error(f"{key}: {type(value)}")
            raise
    
    def _plot_model_weights(self, save_path):
        weights = self.model.get_model_weights()
        plt.figure(figsize=(8, 6))
        model_names = ['Xception', 'Res2Net101', 'EfficientNet-B7']
        sns.barplot(x=model_names, y=weights)
        plt.title('Architecture Contribution Weights')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 

    def evaluate_model(self, model, loader, experiment_name):
        """Evaluate model on given dataset"""
        logger = logging.getLogger(__name__)
        logger.info(f"\nEvaluating {experiment_name}")
        
        # Use cuda if available, even without trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        # Add progress bar for evaluation
        eval_pbar = tqdm(loader, desc='Evaluating', leave=True)
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images = images.to(device)
                outputs = model(images)
                
                # Handle both single model and ensemble outputs
                if isinstance(outputs, torch.Tensor) and outputs.shape[1] == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                else:
                    probs = outputs.squeeze()
                
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.numpy())
                
                # Update progress bar with current batch size
                eval_pbar.set_postfix({
                    'batch_size': images.size(0),
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                })
        
        # Calculate metrics
        try:
            # Convert lists to numpy arrays
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            
            # Calculate all metrics
            auc = roc_auc_score(true_labels, predictions)
            pred_labels = (predictions > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
            accuracy = accuracy_score(true_labels, pred_labels)
            conf_matrix = confusion_matrix(true_labels, pred_labels)
            
            metrics = {
                'performance_metrics': {
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                },
                'raw_data': {
                    'predictions': predictions.tolist(),
                    'true_labels': true_labels.tolist()
                },
                'confusion_matrix': conf_matrix.tolist(),
                'experiment_info': {
                    'name': experiment_name,
                    'num_samples': len(true_labels),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Log results
            logger.info(f"\nResults for {experiment_name}:")
            logger.info(f"{'='*50}")
            logger.info(f"AUC:       {auc:.4f}")
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"F1 Score:  {f1:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"{'='*50}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.error(f"Predictions shape: {len(predictions)}")
            logger.error(f"Labels shape: {len(true_labels)}")
            raise 