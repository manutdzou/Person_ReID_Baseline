import os
import time
import torch
import numpy as np
from torch.autograd import Variable
import models
from config import cfg
from data_loader import data_loader
from loss import make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler
from logger import make_logger
from evaluation import evaluation
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def train(config_file):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    
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
    loss_fn = make_loss(cfg)


    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = torch.device(cfg.MODEL.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger.info("Start training")

    since = time.time()
    for epoch in tqdm(range(epochs), desc='Epoch'):
        count = 0
        running_loss = 0.0
        running_acc = 0
        for data in tqdm(train_loader, desc='Iteration', leave=False):
            model.train()
            images, labels = data

            if device:
                model.to(device)
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            scores, feats = model(images)
            loss = loss_fn(scores, feats, labels)

            loss.backward()
            optimizer.step()

            count = count + 1
            running_loss += loss.item()
            running_acc += (scores.max(1)[1] == labels).float().mean().item()

            if count % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch+1, count, len(train_loader),
                                    running_loss/count, running_acc/count,
                                    scheduler.get_lr()[0]))

        if epoch % checkpoint_period == 0:
            model.save(output_dir,epoch+1)

        # Validation
        if epoch % eval_period == 0:
            all_feats = []
            all_pids = []
            all_camids = []

            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data

                    if device:
                        model.to(device)
                        images = images.to(device)

                    feats = model(images)

                all_feats.append(feats)
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            all_feats = torch.cat(all_feats, dim=0)
            # query
            qf = all_feats[:num_query]
            q_pids = np.asarray(all_pids[:num_query])
            q_camids = np.asarray(all_camids[:num_query])
            # gallery
            gf = all_feats[num_query:]
            g_pids = np.asarray(all_pids[num_query:])
            g_camids = np.asarray(all_camids[num_query:])

            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            cmc, mAP = evaluation(distmat, q_pids, g_pids, q_camids, g_camids)
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('-' * 10)


if __name__=='__main__':
    import fire
    fire.Fire()
