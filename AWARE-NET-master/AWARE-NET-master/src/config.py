from pathlib import Path
import logging
import torch

class Config:
    # Paths
    ROOT_DIR = Path("E:/DeepScan/deepfake_cluster/test-dlib-face")
    DATA_DIR = ROOT_DIR / "faces"
    RESULTS_DIR = ROOT_DIR / "results"
    WEIGHTS_DIR = RESULTS_DIR / "weights"
    ANNOTATIONS_DIR = ROOT_DIR / "annotations"
    LOG_DIR = ROOT_DIR / "logs"
    
    # Dataset structure
    DATASET_STRUCTURE = {
        'ff++': {
            'real_dirs': {
                'actors': 'ff++/real/actors',
                'youtube': 'ff++/real/youtube'
            },
            'fake_dirs': {
                'Deepfakes': 'ff++/fake/Deepfakes',
                'Face2Face': 'ff++/fake/Face2Face',
                'FaceSwap': 'ff++/fake/FaceSwap',
                'NeuralTextures': 'ff++/fake/NeuralTextures',
                'FaceShifter': 'ff++/fake/FaceShifter',
                'DeepFakeDetection': 'ff++/fake/DeepFakeDetection'
            }
        },
        'celebdf': {
            'real_dir': 'celebdf/real',
            'fake_dir': 'celebdf/fake'
        }
    }
    
    # Model names mapping - Single source of truth
    MODELS = {
        'xception': {
            'timm_name': 'legacy_xception',     # name used for timm model creation
            'dir_name': 'xception',             # name used for directory structure
            'variant_key': 'legacy_xception',          # name used in variant names
            'weights_dir': 'xception'    # directory name for weights
        },
        'res2net101_26w_4s': {
            'timm_name': 'res2net101_26w_4s',
            'dir_name': 'res2net101',
            'variant_key': 'res2net101_26w_4s',
            'weights_dir': 'res2net101'
        },
        'tf_efficientnet_b7_ns': {
            'timm_name': 'tf_efficientnet_b7_ns',
            'dir_name': 'efficientnet_b7',
            'variant_key': 'tf_efficientnet_b7_ns',
            'weights_dir': 'efficientnet_b7'
        },
        'ensemble': {
            'timm_name': 'ensemble',
            'dir_name': 'ensemble',
            'variant_key': 'ensemble',
            'weights_dir': 'ensemble'
        }
    }

    @classmethod
    def get_model_weights_dir(cls, model_key, dataset, variant='no_aug'):
        """Get the weights directory for a specific model configuration
        Args:
            model_key (str): Key from MODELS dict ('xception', 'res2net101_26w_4s', etc.)
            dataset (str): Dataset name ('ff++' or 'celebdf')
            variant (str): Training variant ('no_aug' or 'with_aug')
        Returns:
            Path: Directory path for model weights
        """
        if model_key not in cls.MODELS:
            raise ValueError(f"Invalid model key: {model_key}")
        
        weights_dir = cls.WEIGHTS_DIR / dataset / cls.MODELS[model_key]['weights_dir'] / variant
        return weights_dir

    @classmethod
    def get_latest_weights_path(cls, model_key, dataset, variant='no_aug'):
        """Get the path to the latest weights file for a specific model
        Args:
            model_key (str): Key from MODELS dict
            dataset (str): Dataset name
            variant (str): Training variant
        Returns:
            Path: Path to the latest weights file
        """
        weights_dir = cls.get_model_weights_dir(model_key, dataset, variant)
        if not weights_dir.exists():
            return None
            
        weight_files = list(weights_dir.glob('*.pth'))
        if not weight_files:
            return None
            
        # Sort by modification time to get the latest
        return max(weight_files, key=lambda p: p.stat().st_mtime)

    # Create all necessary directories recursively
    @classmethod
    def create_directories(cls):
        # Main directories
        cls.ROOT_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.ANNOTATIONS_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        
        # Setup logging first
        cls.setup_logging()
        logger = logging.getLogger(__name__)
        
        # Create complete directory structure for each dataset and model
        for dataset in ['ff++', 'celebdf']:
            for model_key, model_config in cls.MODELS.items():
                weights_dir = model_config['weights_dir']
                for variant in ['no_aug', 'with_aug']:
                    # Weights directory using timm model name
                    (cls.WEIGHTS_DIR / dataset / weights_dir / variant).mkdir(parents=True, exist_ok=True)
                    
                    # Metrics directory using dir_name for consistency
                    (cls.RESULTS_DIR / 'metrics' / dataset / model_config['dir_name'] / variant).mkdir(parents=True, exist_ok=True)
                    
                    # Plots directory
                    plots_dir = cls.RESULTS_DIR / 'plots' / dataset / model_config['dir_name'] / variant
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    if variant == 'with_aug':
                        (plots_dir / 'augmented_samples').mkdir(exist_ok=True)
        
        # Create cross-evaluation directories
        for source_dataset in ['ff++', 'celebdf']:
            for model_config in cls.MODELS.values():
                dir_name = model_config['dir_name']  # Use directory name from config
                for target_dataset in ['ff++', 'celebdf']:
                    if source_dataset != target_dataset:
                        (cls.RESULTS_DIR / 'cross_evaluation' / source_dataset / dir_name / target_dataset).mkdir(parents=True, exist_ok=True)
        
        # Log created directory structure
        logger.info("Created directory structure with models:")
        for model_config in cls.MODELS.values():
            logger.info(f"- {model_config['dir_name']}")
        
        # Create annotations directory only if needed
        if cls.ANNOTATION_CACHE:
            cls.ANNOTATIONS_DIR.mkdir(exist_ok=True)
            logger.info("Created annotations directory for caching")
    
    # Training params
    BATCH_SIZE = 32
    MAX_EPOCHS = 2
    PATIENCE = 5
    MIN_EPOCHS = 5
    VALIDATION_FREQ = 1
    
    # Early stopping controls
    EARLY_STOPPING = True
    MIN_DELTA = 0.001
    
    # Validation controls
    VALIDATION_METRIC = 'loss'
    VAL_CHECK_INTERVAL = 100
    
    # Learning rate scheduling
    LR_INITIAL = 3e-4
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 2
    LR_SCHEDULE_TYPE = 'cosine'
    
    # Model configs
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    
    # Dataset configs
    DATASET_FRACTION = 0.1
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    IMAGE_SIZE = 224 
    
    # GPU optimization
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    MIXED_PRECISION = True  # Use automatic mixed precision
    NUM_WORKERS = 8  # Usually set to number of CPU cores
    
    # Augmentation configs
    AUGMENTATION_RATIO = 0.3  # 30% increase in dataset size
    AUGMENTATION_PARAMS = {
        # Geometric transformations
        'rotation': {'probability': 0.5, 'max_left': 15, 'max_right': 15},
        'shear': {'probability': 0.3, 'max_shear_left': 10, 'max_shear_right': 10},
        'flip': {'probability': 0.5},
        'skew': {'probability': 0.3, 'magnitude': 0.3},
        
        # Color transformations
        'color_jitter': {
            'probability': 0.3,
            'brightness': 0.2,  # range: [1-x, 1+x]
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1  # range: [-x, x]
        }
    }
    
    # Annotation configs
    FORCE_NEW_ANNOTATIONS = False  # Whether to force create new annotations
    ANNOTATION_CACHE = True        # Whether to use cached annotations
    
    @classmethod
    def optimize_gpu(cls):
        if torch.cuda.is_available():
            # Set GPU memory allocation
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def setup_logging():
        """Setup basic logging configuration"""
        # Create log directory if it doesn't exist
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
        console_handler.setFormatter(console_format)
        
        # Create file handler
        file_handler = logging.FileHandler(Config.LOG_DIR / 'training.log')
        file_handler.setFormatter(console_format)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)