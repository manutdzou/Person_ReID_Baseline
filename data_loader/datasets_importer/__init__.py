# encoding: utf-8

from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .msmt17 import MSMT17
from .ntucampus import NTUCampus
from .dataset_loader import ImageDataset

__factory = {
    'Market1501': Market1501,
    'DukeMTMC': DukeMTMC,
    'MSMT17': MSMT17,
    'NTUCampus': NTUCampus,
}


def get_names():
    return __factory.keys()


def init_dataset(cfg, *args, **kwargs):
    if cfg.DATASETS.NAMES not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(cfg.DATASETS.NAMES))
    return __factory[cfg.DATASETS.NAMES](cfg,*args, **kwargs)
