"""
Utility module - shared functions, configuration, and training utilities
"""

from .config import *
from .util import *
from .trainer import Trainer

__all__ = [
    'Trainer',
    # config exports
    'SEED', 'BATCH_SIZE', 'DEVICE', 'IMAGE_SIZE',
    'CELEBDF_ROOT', 'CELEBDF_VAL_SPLIT', 'CELEBDF_BALANCE_VIDEOS',
    'get_model_info', 'validate_model_type',
    'get_checkpoint_dir', 'get_model_path', 'get_log_dir', 'get_train_log_path', 'get_test_log_path',
    'get_celebdf_checkpoint_dir', 'get_celebdf_model_path', 'get_celebdf_log_dir',
    'get_celebdf_train_log_path', 'get_celebdf_test_log_path',
    'CROSS_DATASET_TEST_DATASETS', 'DEFAULT_MODEL_TYPE',
    # util exports
    'seed_everything', 'set_seed', 'evaluate', 'save_test_log',
]
