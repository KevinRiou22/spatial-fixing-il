#from .video_multi_view_backup import VideoMultiViewModel as MODEL
from .video_multi_view import VideoMultiViewModel as MODEL
from .fuse_views_mht import fuse_views_mht as MODEL_mht
def get_models(cfg):

    if cfg.NETWORK.TYPE == 'MHT':
        train_model = MODEL_mht(cfg, is_train=True)
        test_model = MODEL_mht(cfg, is_train=False)
    else:
        train_model = MODEL(cfg, is_train=True)
        test_model = MODEL(cfg, is_train=False)
    return train_model, test_model
