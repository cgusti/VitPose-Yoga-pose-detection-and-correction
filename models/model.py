import torch.nn as nn

from .backbone.vit import ViT #custom implementation of Vision Transformer backbone 
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']

class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'} #create a new cfg dict. for backbone cfg
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'} #create new cfg for head cfg 
        
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)  #initialize keypoint head of model 
    
    def forward_features(self, x):
        return self.backbone(x) #pass input x through ViT backbone 
    
    def forward(self, x):
        return self.keypoint_head(self.backbone(x)) #pass input x through output heads 