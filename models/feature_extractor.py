import timm
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class FeatureExtractor(nn.Module):
    def __init__(self, network = 'resnet34'):
        super(FeatureExtractor, self).__init__()
        model = timm.create_model(network, pretrained=True)
        self.num_features = model.num_features
        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.freeze_layers()

    def freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.feature_extractor(x)
        return out
