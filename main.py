from config import cfg # local variable usage pattern, or:# from config import cfg  # global singleton usage pattern
from data_loader import data_loader

def train(config_file):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print(cfg)
    train_loader, val_loader, num_query, num_classes = data_loader(cfg)
    
    model = getattr(models, opt.model)(num_classes)
    model.cuda()



if __name__=='__main__':
    import fire
    fire.Fire()
