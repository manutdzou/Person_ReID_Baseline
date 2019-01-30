import os
import models
from config import cfg
from data_loader import data_loader
from loss import make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler
from logger import make_logger
from tqdm import tqdm

def train(config_file):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    logger = make_logger("reid_baseline", output_dir)
    logger.info("Using {} GPUS".format(1))
    logger.info("Loaded configuration file {}".format(config_file))
    logger.info("Running with config:\n{}".format(cfg))
     
    train_loader, val_loader, num_query, num_classes = data_loader(cfg)

    model = getattr(models, cfg.MODEL.NAME)(num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg,optimizer)
    loss_func = make_loss(cfg)
    # model.cuda()



if __name__=='__main__':
    import fire
    fire.Fire()
