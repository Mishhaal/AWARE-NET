import torch
from torch.utils.data import DataLoader
import logging
import json
from tqdm import tqdm
from pathlib import Path

from model import EnsembleDeepfakeDetector, SingleModelDetector
from dataset import DeepfakeDataset, create_data_splits
from experiments import ExperimentRunner

def cross_dataset_evaluate(config, transform, source_dataset, target_dataset, model_name, model_path):
    """Evaluate model trained on source dataset on target dataset"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\nCross-Dataset Evaluation:")
    logger.info(f"Source Dataset: {source_dataset}")
    logger.info(f"Target Dataset: {target_dataset}")
    logger.info(f"Model: {model_name}")
    
    # Load target dataset
    _, _, test_df = create_data_splits(config, target_dataset, force_new=True)
    
    test_dataset = DeepfakeDataset(
        dataframe=test_df,
        transform=transform,
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load trained model
    if model_name == 'ensemble':
        model = EnsembleDeepfakeDetector(
            config=config,
            dataset=source_dataset,
            augment='with_aug' in str(model_path)
        ).cuda()
    else:
        model = SingleModelDetector(model_name, config).cuda()
    
    # Get the weights directory
    weights_dir = config.WEIGHTS_DIR / source_dataset / model_name / ('no_aug' if 'no_aug' in str(model_path) else 'with_aug')
    
    # Find the weight file (should be only one)
    weight_files = list(weights_dir.glob('*.pth'))
    if not weight_files:
        raise FileNotFoundError(f"No model weights found in {weights_dir}")
    
    model_path = weight_files[0]  # There should be only one file
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create experimenter for evaluation
    experimenter = ExperimentRunner(config, model, None)  # No trainer needed for evaluationm
    
    # Evaluate
    metrics = experimenter.evaluate_model(
        model=model,
        loader=test_loader,
        experiment_name=f"{source_dataset}_to_{target_dataset}_{model_name}"
    )
    
    # Save cross-evaluation results
    results_dir = config.RESULTS_DIR / 'cross_evaluation' / source_dataset / model_name / target_dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"\nCross-Evaluation Results ({source_dataset} -> {target_dataset}):")
    logger.info(json.dumps(metrics['performance_metrics'], indent=2))
    
    return metrics

def run_cross_evaluations(config, transform, models_config):
    """Run all cross-dataset evaluations"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50 + "\nPerforming Cross-Dataset Evaluation\n" + "="*50)
    
    for model_key, model_config in models_config.items():
        model_name = model_config['name'] if 'name' in model_config else model_key
        
        # Check for models in both no_aug and with_aug directories
        for aug_type in ['no_aug', 'with_aug']:
            # FF++ → CelebDF
            weights_dir = config.WEIGHTS_DIR / 'ff++' / model_name / aug_type
            if list(weights_dir.glob('*.pth')):
                cross_dataset_evaluate(config, transform, 'ff++', 'celebdf', model_name, weights_dir)
            
            # CelebDF → FF++
            weights_dir = config.WEIGHTS_DIR / 'celebdf' / model_name / aug_type
            if list(weights_dir.glob('*.pth')):
                cross_dataset_evaluate(config, transform, 'celebdf', 'ff++', model_name, weights_dir)