import torch
import torch.nn as nn
import timm
from pretrainedmodels import xception

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'xception':
            self.model = xception(pretrained='imagenet')
            self.model.last_linear = nn.Linear(2048, 1)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize three instances of each architecture
        self.xception_models = nn.ModuleList([
            BaseModel('xception') for _ in range(3)
        ])
        self.res2net_models = nn.ModuleList([
            BaseModel('res2net101_26w_4s') for _ in range(3)
        ])
        self.efficientnet_models = nn.ModuleList([
            BaseModel('tf_efficientnet_b7_ns') for _ in range(3)
        ])
        
        # Learnable weights for each model type
        self.architecture_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x):
        # Get predictions from each instance of each architecture
        xception_preds = torch.mean(torch.stack([model(x) for model in self.xception_models]), dim=0)
        res2net_preds = torch.mean(torch.stack([model(x) for model in self.res2net_models]), dim=0)
        efficientnet_preds = torch.mean(torch.stack([model(x) for model in self.efficientnet_models]), dim=0)
        
        # Stack predictions from different architectures
        arch_preds = torch.stack([xception_preds, res2net_preds, efficientnet_preds], dim=1)
        
        # Weight the architecture predictions
        weights = torch.softmax(self.architecture_weights, dim=0)
        ensemble_pred = torch.sum(arch_preds * weights.view(1, -1, 1), dim=1)
        
        return ensemble_pred
    
    def get_model_weights(self):
        return torch.softmax(self.architecture_weights, dim=0).detach().cpu().numpy()

ensemble = EnsembleDeepfakeDetector() 