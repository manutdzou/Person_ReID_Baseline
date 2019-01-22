from config import cfg # local variable usage pattern, or:# from config import cfg  # global singleton usage pattern

def train(config_file):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print(cfg)



if __name__=='__main__':
    import fire
    fire.Fire()
