import os
from src import utils



cfg = utils.Config()

###########################################################################
########################### general hyperparams ###########################
###########################################################################
cfg.seed = None
cfg.data_dir = './data'
cfg.models_dir = 'models'
cfg.data_path = os.path.join('data')
cfg.save_log = True                 # additionally save log and training loss logs to a .csv file
cfg.epochs_evaluate_train = 1       # evaluate train (in eval mode with no_grad) every epochs_evaluate_train epochs
cfg.epochs_evaluate_validation = 1  # evaluate validation (in eval mode with no_grad) every epochs_evaluate_validation epochs
cfg.num_workers = 6                 # num_workers for data loader
cfg.epochs_save = 20              # save a checkpoint (additionally to last and best) every epochs_save epochs

cfg.save = True # save model checkpoints (best, last and epoch)
cfg.tqdm_bar = True # using a tqdm bar for loading data and epoch progression, should be False if not using a jupyter notebook
cfg.preload_data = True             # preloading data to memory
cfg.prints = 'display' # should be 'display' if using a jupyter notebook, else 'print'
cfg.load = 1
cfg.max_iterations = None
cfg.wd = 5e-4


###########################################################################
############################ model hyperparams ############################
###########################################################################
cfg.backbone = 'resnet50'
cfg.bs = 64  # 32 96 64
cfg.epochs = 600  # 600 800 1000

cfg.num_classes = 10

cfg.optimizer = 'sgd'  # adam sgd
cfg.optimizer_params = {}
cfg.optimizer_momentum = 0.9
cfg.lr = 0.01  # 3e-4 1e-3
cfg.min_lr = 5e-8
cfg.best_policy = 'val_score'
cfg.bias = True
cfg.version = f'{cfg.backbone}_{cfg.optimizer}_bs{cfg.bs}'
