from beans.beats.BEATs import BEATs, BEATsConfig
import torch 
import torch.nn as nn

class BEATsClassifier(nn.Module):
    def __init__(self, model_path, num_classes, multi_label=False):
        super().__init__()
        checkpoint = torch.load(model_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])

        self.model.finetuned_model = False
        self.model.predictor_class = num_classes

        self.model.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
        self.model.predictor = nn.Linear(cfg.encoder_embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor, mixup_lambda = None, infer_mode = False, device=None):
        sample = {}
        #print(x.keys())
        x = x["waveform"].to(device=device, non_blocking=True)
        if torch.isnan(x).any():
            mask = torch.isnan(x)
            x = torch.nan_to_num(x)
        else:
            mask = torch.zeros(x.shape).bool()
        
        sample["embedding"] = self.model.extract_features(x, mask)[0]
        return sample



def create_beats_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    return BEATsClassifier("/cluster/work/boraa/beans/model_weights/beats.pth", audio_cfg.class_num)
    