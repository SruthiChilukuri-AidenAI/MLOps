import os
from omegaconf import OmegaConf
import pathlib

def get_mlflow_cfg():
    p = os.path.abspath(__file__)
    #print(p)
    path = os.sep.join(p.split(os.sep)[:-1])
    #print(path)
    cfg_file = os.path.join(path,'mlflow_cfg.yaml')
    #print(cfg_file)
    config = OmegaConf.load(cfg_file)
    return config

if __name__ == "__main__":
    get_mlflow_cfg()
