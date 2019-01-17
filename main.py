
from config import cfg # local variable usage pattern, or:# from config import cfg  # global singleton usage pattern

cfg.merge_from_file("./config/softmax_triplet.yaml")
cfg.freeze()
print(cfg)
print(cfg.DATALOADER.NUM_WORKERS)
