from .beats_model.BEATs import BEATs, BEATsConfig
import torch 
import torch.nn as nn

class BEATsClassifier(nn.Module): # Name does not make sense, it is more like encoder/backbone
    def __init__(self, model_path):
        super().__init__()
        checkpoint = torch.load(model_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])

        self.sap = SelfAttentionPooling(496)
    
    def forward(self, x: torch.Tensor, mixup_lambda = None, infer_mode = False, device=None):
        sample = {}
        #print(x.keys())
        x = x["waveform"].to(device=device, non_blocking=True)
        if torch.isnan(x).any():
            mask = torch.isnan(x)
            x = torch.nan_to_num(x)
        else:
            mask = torch.zeros(x.shape).bool()
        
        #print(x.shape)
        embedding = self.model.extract_features(x, mask)[0]
        sample["embedding"] = torch.mean(embedding, dim = 1) 
        return sample



def create_beats_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    return BEATsClassifier("/cluster/work/boraa/beans/model_weights/beats.pth")
    