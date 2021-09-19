import os
from src import utils



cfg = utils.Config()

###########################################################################
########################### general hyperparams ###########################
###########################################################################
cfg.seed = None
cfg.models_dir = 'models'
cfg.data_path = 'data'
cfg.save_log = True                 # additionally save log and training loss logs to a .csv file
cfg.epochs_evaluate_train = 1       # evaluate train (in eval mode with no_grad) every epochs_evaluate_train epochs
cfg.epochs_evaluate_validation = 1  # evaluate validation (in eval mode with no_grad) every epochs_evaluate_validation epochs
cfg.num_workers = 2                 # num_workers for data loader
cfg.epochs_save = 20                # save a checkpoint (additionally to last and best) every epochs_save epochs

cfg.save = True # save model checkpoints (best, last and epoch)
cfg.tqdm_bar = False # using a tqdm bar for loading data and epoch progression, should be False if not using a jupyter notebook
cfg.prints = 'print' # should be 'display' if using a jupyter notebook, else 'print'
cfg.load = None
cfg.max_iterations = None
cfg.wd = 0 #  5e-4


###########################################################################
############################ model hyperparams ############################
###########################################################################
cfg.backbone = 'resnet34'
cfg.bs = 32  # 32 96 64
cfg.epochs = 100  # 600 800 1000

cfg.optimizer = 'adam'  # adam sgd
cfg.optimizer_params = {}
cfg.optimizer_momentum = 0.9
cfg.lr = 3e-4  # 3e-4 1e-3
cfg.min_lr = 5e-8
cfg.best_policy = 'val_score'
cfg.bias = True
cfg.cos = False

# cfg.feature_extraction = True
cfg.feature_extraction = False
cfg.rotation = True
if cfg.rotation:
    cfg.model_type = 'rotation'
    cfg.angles = tuple(range(0, 100, 10))
else:
    cfg.model_type = 'no_rotation'
    cfg.angles = (0, )

cfg.version = f'{cfg.model_type}{"_extracted" if cfg.feature_extraction else ""}_{cfg.backbone}_{cfg.optimizer}_lr{cfg.lr}_bs{cfg.bs}{"_cos" if cfg.cos else ""}'

