import torch
from torch.utils.data import DataLoader
from .transforms import transforms
from .datasets_importer import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler

def data_loader(cfg,dataset_name):
    train_transforms = transforms(cfg, is_train=True)
    val_transforms = transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg,dataset_name)
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids