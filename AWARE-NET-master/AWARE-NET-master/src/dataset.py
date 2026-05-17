import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import Augmentor
import json
import logging
import matplotlib.pyplot as plt

class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, transform=None, augment=False, variant_name=None, config=None):
        self.data = dataframe
        self.transform = transform
        self.augment = augment
        self.variant_name = variant_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if augment:
            self.logger.info("Initializing data augmentation pipeline:")
            self.augmentor = Augmentor.Pipeline(".")
            
            # Geometric transformations
            params = self.config.AUGMENTATION_PARAMS
            
            self.logger.info(f"- Rotation (±{params['rotation']['max_left']}°): p={params['rotation']['probability']}")
            self.augmentor.rotate(
                probability=params['rotation']['probability'],
                max_left_rotation=params['rotation']['max_left'],
                max_right_rotation=params['rotation']['max_right']
            )
            
            self.logger.info(f"- Shear (±{params['shear']['max_shear_left']}°): p={params['shear']['probability']}")
            self.augmentor.shear(
                probability=params['shear']['probability'],
                max_shear_left=params['shear']['max_shear_left'],
                max_shear_right=params['shear']['max_shear_right']
            )
            
            self.logger.info(f"- Flip: p={params['flip']['probability']}")
            self.augmentor.flip_left_right(probability=params['flip']['probability'])
            
            self.logger.info(f"- Skew: p={params['skew']['probability']}")
            self.augmentor.skew(
                probability=params['skew']['probability'],
                magnitude=params['skew']['magnitude']
            )
            
            # Color transformations
            color_params = params['color_jitter']
            self.logger.info(f"- Color Jitter: p={color_params['probability']}")
            self.augmentor.random_brightness(
                probability=color_params['probability'],
                min_factor=1-color_params['brightness'],
                max_factor=1+color_params['brightness']
            )
            self.augmentor.random_contrast(
                probability=color_params['probability'],
                min_factor=1-color_params['contrast'],
                max_factor=1+color_params['contrast']
            )
            self.augmentor.random_color(
                probability=color_params['probability'],
                min_factor=1-color_params['saturation'],
                max_factor=1+color_params['saturation']
            )
            
            # Create augmented versions of the dataset
            self.augmented_data = self._create_augmented_dataset()
            self.data = pd.concat([self.data, self.augmented_data], ignore_index=True)
            self.logger.info(f"Dataset size increased from {len(dataframe)} to {len(self.data)} with augmentations")
            
            # Save sample augmented images
            self.save_augmented_samples()
    
    def _create_augmented_dataset(self):
        """Create augmented versions of the dataset"""
        augmented_data = []
        
        # Calculate how many augmented images we need
        num_images = len(self.data)
        target_aug_images = int(num_images * self.config.AUGMENTATION_RATIO)
        
        self.logger.info(f"Creating {target_aug_images} augmented images ({self.config.AUGMENTATION_RATIO*100}% of {num_images})")
        
        # Randomly select images to augment
        indices = np.random.choice(num_images, size=target_aug_images, replace=True)
        
        for idx in indices:
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            
            # Create augmented version
            image_np = np.array(image)
            aug_image_np = self.augmentor._execute_with_array(image_np)
            
            # Create new row with augmented image info
            aug_row = row.copy()
            aug_row['is_augmented'] = True
            augmented_data.append(aug_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        self.logger.info(f"Created {len(augmented_df)} augmented images")
        
        return augmented_df
    
    def save_augmented_samples(self):
        """Save sample augmented images for visualization"""
        try:
            # Randomly select 3 images
            sample_indices = np.random.choice(len(self.data), size=3, replace=False)
            
            for i, idx in enumerate(sample_indices):
                row = self.data.iloc[idx]
                orig_image = Image.open(row['image_path']).convert('RGB')
                
                # Create 3x3 grid of augmented images
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle('Deepfake-Specific Augmentations', fontsize=16)
                
                # Original image in center
                axes[1, 1].imshow(orig_image)
                axes[1, 1].set_title('Original')
                axes[1, 1].axis('off')
                
                # 8 augmented versions around it
                aug_positions = [(i,j) for i in range(3) for j in range(3) if (i,j) != (1,1)]
                for pos in aug_positions:
                    aug_image = orig_image.copy()
                    aug_image_np = np.array(aug_image)
                    aug_image_np = self.augmentor._execute_with_array(aug_image_np)
                    aug_image = Image.fromarray(aug_image_np.astype('uint8'))
                    
                    axes[pos[0], pos[1]].imshow(aug_image)
                    axes[pos[0], pos[1]].axis('off')
                
                # Save plot
                dataset = 'ff++' if 'ff++' in row['image_path'] else 'celebdf'
                model_type = self.variant_name.split('_')[1]
                save_dir = self.config.RESULTS_DIR / 'plots' / dataset / model_type / 'with_aug' / 'augmented_samples'
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f'augmented_grid_{i+1}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved augmented sample grid to {save_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving augmented samples: {str(e)}")
            self.logger.error(f"Error details: {str(e.__class__.__name__)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        
        if self.augment:
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            # Apply augmentations
            image_np = self.augmentor._execute_with_array(image_np)
            # Convert back to PIL Image
            image = Image.fromarray(image_np.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
            
        return image, row['label']

def create_data_splits(config, dataset_name, force_new=None):
    logger = logging.getLogger(__name__)
    
    # Check if we should use force_new from config
    if force_new is None:
        force_new = config.FORCE_NEW_ANNOTATIONS
    
    # Check for existing annotations if caching is enabled
    if config.ANNOTATION_CACHE:
        annotation_file = config.ANNOTATIONS_DIR / f'{dataset_name}_splits.json'
        
        if annotation_file.exists() and not force_new:
            logger.info(f"Loading cached {dataset_name} annotations...")
            with open(annotation_file, 'r') as f:
                splits = json.load(f)
                return pd.DataFrame(splits['train']), pd.DataFrame(splits['val']), pd.DataFrame(splits['test'])
    
    logger.info(f"Creating new annotations for {dataset_name}")
    logger.info(f"Starting data loading process for {dataset_name} dataset")
    logger.info(f"Looking for data in {config.DATA_DIR}")
    logger.info(f"Using {config.DATASET_FRACTION * 100}% of total data")
    
    all_data = []
    
    if dataset_name == 'ff++':
        # Process FF++ dataset
        # Real images
        for real_source, real_path in config.DATASET_STRUCTURE['ff++']['real_dirs'].items():
            real_dir = config.DATA_DIR / real_path
            logger.debug(f"Looking for FF++ real images in: {real_dir}")
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
            logger.info(f"Found {len(real_paths)} real FF++ images in {real_source}")
            
            all_data.extend([{
                'image_path': str(p),
                'label': 0,
                'subset': f'real_{real_source}',
                'dataset': 'ff++'
            } for p in real_paths[:int(len(real_paths) * config.DATASET_FRACTION)]])
        
        # Fake images
        for fake_type, fake_path in config.DATASET_STRUCTURE['ff++']['fake_dirs'].items():
            fake_dir = config.DATA_DIR / fake_path
            logger.debug(f"Looking for {fake_type} fake images in: {fake_dir}")
            fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
            logger.info(f"Found {len(fake_paths)} fake images in {fake_type}")
            
            all_data.extend([{
                'image_path': str(p),
                'label': 1,
                'subset': fake_type,
                'dataset': 'ff++'
            } for p in fake_paths[:int(len(fake_paths) * config.DATASET_FRACTION)]])
            
    else:  # celebdf
        # Real images
        real_dir = config.DATA_DIR / config.DATASET_STRUCTURE['celebdf']['real_dir']
        real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        logger.info(f"Found {len(real_paths)} real CelebDF images")
        
        all_data.extend([{
            'image_path': str(p),
            'label': 0,
            'dataset': 'celebdf'
        } for p in real_paths[:int(len(real_paths) * config.DATASET_FRACTION)]])
        
        # Fake images
        fake_dir = config.DATA_DIR / config.DATASET_STRUCTURE['celebdf']['fake_dir']
        fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
        logger.info(f"Found {len(fake_paths)} fake CelebDF images")
        
        all_data.extend([{
            'image_path': str(p),
            'label': 1,
            'dataset': 'celebdf'
        } for p in fake_paths[:int(len(fake_paths) * config.DATASET_FRACTION)]])
    
    if not all_data:
        raise ValueError(f"No images found for {dataset_name}")
    
    # Create DataFrame and split
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_idx = int(len(df) * config.TRAIN_SPLIT)
    val_idx = train_idx + int(len(df) * config.VAL_SPLIT)
    
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Real samples: {len(df[df['label'] == 0])}")
    logger.info(f"Fake samples: {len(df[df['label'] == 1])}")
    if 'subset' in df.columns:
        for subset in df['subset'].unique():
            logger.info(f"Samples in {subset}: {len(df[df['subset'] == subset])}")
    
    logger.info(f"Data Split Details:")
    logger.info(f"Training set: {len(train_df)} samples ({len(train_df[train_df['label'] == 0])} real, {len(train_df[train_df['label'] == 1])} fake)")
    logger.info(f"Validation set: {len(val_df)} samples ({len(val_df[val_df['label'] == 0])} real, {len(val_df[val_df['label'] == 1])} fake)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df[test_df['label'] == 0])} real, {len(test_df[test_df['label'] == 1])} fake)")
    
    # Save annotations if caching is enabled
    if config.ANNOTATION_CACHE:
        splits = {
            'train': train_df.to_dict('records'),
            'val': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(splits, f, indent=4)
        logger.info(f"Cached annotations to {annotation_file}")
    
    return train_df, val_df, test_df 