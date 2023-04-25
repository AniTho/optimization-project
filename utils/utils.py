import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import wandb
from models.feature_extractor import FeatureExtractor

def save_checkpoint(model, save_path = 'saved_models/model.pt'):
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, save_path = 'saved_models/model.pt'):
    model.load_state_dict(torch.load(save_path))
    return model

def determinantal_point_process(X, device = 'cpu'):
    extractor = FeatureExtractor(network='resnet50').to(device)
    inv_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.5, 1/0.5, 1/0.5]),
                                      transforms.Normalize(mean = [-0.5, -0.5, -0.5],
                                         std = [ 1., 1., 1. ])])
    normalize_imagenet = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225])])
    X = normalize_imagenet(inv_transform(X))
    features = extractor(X)
    similarity = torch.matmul(features, features.T)
    eigen_vals, _ = torch.linalg.eigh(similarity)
    dpp = torch.sum(torch.log(eigen_vals))
    return dpp

def visualization(model, title, latent_dim, clear = True, device = 'cpu'):
    inv_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.5, 1/0.5, 1/0.5]),
                                      transforms.Normalize(mean = [-0.5, -0.5, -0.5],
                                         std = [ 1., 1., 1. ])])
    model.eval()
    fig_1, axes_1 = plt.subplots(nrows = 6, ncols = 6, figsize = (6,6))
    with torch.no_grad():
        latent_vec = torch.randn(36, latent_dim).to(device)
        imgs = inv_transform(model(latent_vec))
        for idx in range(36):
            img = np.transpose(imgs[idx].detach().cpu().numpy(), (1,2,0))
            axes_1[idx//6, idx%6].imshow(img)
            axes_1[idx//6, idx%6].axis('off')
    wandb.log({f"{title}_regenerated":fig_1})
    if clear:
        fig_1.clear()
        plt.close(fig_1)