from omegaconf import OmegaConf

class Config:
    def __init__(self, path: str):
        self.cfg = OmegaConf.load(path)
    
    def __getattr__(self, name):
        """Access nested config values using dot notation"""
        return OmegaConf.select(self.cfg, name)
    
    @property
    def vitals(self):
        return self.cfg.agent
    
    @property
    def world(self):
        return self.cfg.world
    
    @property
    def training(self):
        return self.cfg.training

config = Config('configs/default.yaml')