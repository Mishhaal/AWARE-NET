import torch
import torch.nn as nn
import timm
import logging
import os

class BaseModel(nn.Module):
    def __init__(self, timm_name, config, pretrained=True):
        super().__init__()
        self.timm_name = timm_name
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create model with 2 output classes (real/fake)
            self.model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=2,  # Binary classification
                drop_rate=config.DROPOUT_RATE   # Add dropout for regularization
            )
            
            # Log model size
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Loaded {timm_name} with {num_params:,} parameters")
            
        except Exception as e:
            self.logger.error(f"Error loading {timm_name}: {str(e)}")
            raise
    
    def forward(self, x):
        return self.model(x)
    
    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights, handling both full checkpoints and state dicts"""
        try:
            checkpoint = torch.load(weights_path)
            # If it's a full checkpoint with metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            # If it's just the model state dict
            else:
                self.model.load_state_dict(checkpoint)
            return True
        except Exception as e:
            self.logger.warning(f"No pre-trained weights found for {self.timm_name}, using base initialization")
            return False

class SingleModelDetector(nn.Module):
    def __init__(self, model_name, config, dataset=None, augment=False, variant_name=None):
        super().__init__()
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Get the actual timm model name from config
        timm_model_name = config.MODELS[model_name]['timm_name']
        self.model = BaseModel(timm_model_name, config)
        self.timm_name = model_name  # Store the config key instead of timm_name
    
    def forward(self, x):
        return self.model(x)
    
    def get_model_weights(self):
        # For single models, return None or empty array since there are no ensemble weights
        return None
    
    def load_weights(self, weights_dir):
        weights_files = list(weights_dir.glob('*.pth'))
        
        if weights_files:
            # Get the latest weights file
            weights_path = sorted(weights_files, key=lambda x: float(x.stem.split('_')[-1]))[0]
            self.logger.info(f"Found weights at: {weights_path}")
            
            try:
                # Load the checkpoint
                checkpoint = torch.load(weights_path, map_location='cpu')
                
                # Extract model state dict from checkpoint
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                    
                # Create new state dict with corrected keys
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the extra 'model.' prefix if present
                    if key.startswith('model.model.'):
                        new_key = key.replace('model.model.', 'model.')
                    elif key.startswith('model.'):
                        new_key = key
                    else:
                        new_key = f'model.{key}'
                    new_state_dict[new_key] = value
                
                # Load the state dict
                try:
                    self.model.load_state_dict(new_state_dict)
                    self.logger.info(f"Successfully loaded weights for {self.timm_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load weights for {self.timm_name}: {str(e)}")
                    # Try loading with strict=False as fallback
                    self.model.load_state_dict(new_state_dict, strict=False)
                    self.logger.warning(f"Loaded weights with strict=False for {self.timm_name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing weights for {self.timm_name}: {str(e)}")
                raise
        else:
            self.logger.warning(f"No pretrained weights found for {self.timm_name}")

class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config, dataset=None, augment=False, variant_name=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.augment = augment
        self.dataset = dataset
        self.variant_name = variant_name
        
        # Initialize individual models
        self.models = self._initialize_models(config)
        
        # Initialize model weights (equal weighting by default)
        num_models = len(self.models)
        if num_models > 0:
            self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:
            self.logger.error("No models were initialized for the ensemble")
            raise ValueError("No models were initialized for the ensemble")
        
        # Create ensemble weights directory
        ensemble_weights_dir = config.get_model_weights_dir('ensemble', dataset, 'with_aug' if augment else 'no_aug')
        ensemble_weights_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_weights_path(self, config, dir_name):
        """Get path to model weights based on augmentation setting and dataset"""
        # Use the new config helper method
        weights_path = config.get_latest_weights_path(
            model_key=next(k for k, v in config.MODELS.items() if v['weights_dir'] == dir_name),
            dataset=self.dataset,
            variant='with_aug' if self.augment else 'no_aug'
        )
        
        if weights_path is None:
            self.logger.warning(f"No weights found for {dir_name} in {self.dataset}")
            return None
            
        self.logger.info(f"Found weights at: {weights_path}")
        return weights_path
    
    def _initialize_models(self, config):
        """Initialize individual models for the ensemble."""
        models = nn.ModuleList()
        
        for model_name, model_config in config.MODELS.items():
            try:
                # Skip ensemble and disabled models
                if model_name == 'ensemble' or not model_config.get('enabled', True):
                    continue
                
                # Create single model detector
                model = SingleModelDetector(
                    model_name=model_name,  # Pass the model name correctly
                    config=config,
                    dataset=self.dataset,
                    augment=self.augment,
                    variant_name=self.variant_name
                )
                
                # Load pre-trained weights if available
                weights_dir = config.get_model_weights_dir(model_name, self.dataset, 'with_aug' if self.augment else 'no_aug')
                if weights_dir.exists():
                    try:
                        model.load_weights(weights_dir)
                        self.logger.info(f"Loaded pre-trained weights for {model_name} from {weights_dir}")
                    except Exception as e:
                        self.logger.error(f"Failed to load weights for {model_name}: {str(e)}")
                else:
                    self.logger.warning(f"No weights directory found for {model_name} at {weights_dir}")
                
                models.append(model)
                self.logger.info(f"Added {model_name} to ensemble")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {model_name}: {str(e)}")
                continue
        
        if len(models) == 0:
            raise ValueError("No models were successfully initialized for the ensemble")
        
        return models
    
    def _load_model_weights(self, model, weights_path, model_name):
        """Load pre-trained weights into model"""
        self.logger.info(f"Loading weights for {model_name} from {weights_path.name}")
        
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle both old and new checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.logger.info(f"Loading checkpoint from epoch {checkpoint['epoch']} "
                               f"with validation loss {checkpoint['val_loss']:.4f}")
            else:
                state_dict = checkpoint
            
            # Fix state dict keys if needed
            if any(k.startswith('model.model.') for k in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.model.'):
                        new_key = key.replace('model.model.', 'model.')
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Load weights
            model.load_state_dict(state_dict)
            self.logger.info(f"Successfully loaded weights for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error loading weights for {model_name}: {str(e)}")
            raise
    
    def forward(self, x):
        """Forward pass using weighted average ensemble"""
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)  # [batch_size, 2]
            pred = torch.softmax(pred, dim=1)  # Convert to probabilities
            predictions.append(pred)
        
        # Stack predictions: [batch_size, num_models, 2]
        stacked_preds = torch.stack(predictions, dim=1)
        
        # Apply learned weights
        weights = torch.softmax(self.model_weights, dim=0)  # Ensures weights sum to 1
        weighted_preds = stacked_preds * weights.view(1, -1, 1)
        
        # Average predictions
        ensemble_output = weighted_preds.sum(dim=1)  # [batch_size, 2]
        
        return ensemble_output
    
    def get_model_weights(self):
        """Get normalized weights for visualization"""
        with torch.no_grad():
            weights = torch.softmax(self.model_weights, dim=0)
            return weights.cpu().numpy()