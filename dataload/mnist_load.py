from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
from PIL import Image
from models.feature_extractor import FeatureExtractor
from submodlib.functions import FacilityLocationFunction
from submodlib.functions import DisparityMinFunction
from submodlib.functions import DisparitySumFunction
from submodlib.functions import LogDeterminantFunction
from tqdm import tqdm
import numpy as np
import random
import os

class MNISTDataset(Dataset):
    def __init__(self, subset = False, sub_selection_technique = None,  percentage = 0.1, 
                 transform = None, bs = 64):
        mnist = datasets.MNIST(root = 'data/', train = True, download=True)
        self.data = mnist.data
        if not subset:
            self.idxs = list(range(len(self.data)))
        else:
            # For storing idxs in a file and loading for fast retrieval
            if os.path.exists(f'data/mnist_{sub_selection_technique}_{int(percentage*100)}.txt'):
                self.idxs = np.loadtxt(f'data/mnist_{sub_selection_technique}_{int(percentage*100)}.txt').tolist()
            else:
                self.idxs = self.subset_selection(sub_selection_technique, percentage, bs)
                np.savetxt(f'data/mnist_{sub_selection_technique}_{int(percentage*100)}.txt', np.array(self.idxs), fmt='%d')
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img = self.data[self.idxs[idx]]
        # Convert to 3 channel image
        img = torch.permute(torch.stack([img, img, img]), (1,2,0)).numpy()
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

    def selection(self, gset, sub_selection_technique, percentage):
        budget = int(percentage*len(gset))
        n_ground = len(gset)
        if sub_selection_technique == 'facility_location':
            sub_selection_method = FacilityLocationFunction(n = n_ground, mode = 'dense', data = gset, metric='euclidean')
        elif sub_selection_technique == 'disparity_min':
            sub_selection_method = DisparityMinFunction(n = n_ground, mode = 'dense', data = gset, metric='euclidean')
        elif sub_selection_technique == 'disparity_sum':
            sub_selection_method = DisparitySumFunction(n = n_ground, mode = 'dense', data = gset, metric='euclidean')
        elif sub_selection_technique == 'log_determinant':
            sub_selection_method = LogDeterminantFunction(n = n_ground, mode = 'dense', lambdaVal=1, data = gset, metric='euclidean')
        else:
            raise Exception("Mentioned method for subset selection is not available")
        
        list_of_idxs = sub_selection_method.maximize(budget, optimizer='StochasticGreedy', show_progress = False)
        idxs = [x[0] for x in list_of_idxs]
        return idxs
    
    def subset_selection(self, sub_selection_technique, percentage, bs):
        budget = int(percentage*len(self.data))
        print(f'{"*"*20} Selecting Subset {"*"*20}')
        if sub_selection_technique == 'random':
            all_idxs = list(range(len(self.data)))
            idxs = random.sample(all_idxs, budget)
            return idxs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.RandomCrop(64, 64),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])
                                        ])
        feature_extractor = FeatureExtractor('resnet50')
        feature_extractor = feature_extractor.cuda(1)
        feature_extractor.eval()
        final_idxs = []
        with torch.no_grad():
            print('Forming Batches for generating ground set')
            complete_dataset = MNISTDataset(transform = transform)
            dataloader = DataLoader(complete_dataset, batch_size = bs, shuffle=False)
            list_of_features = []
            for idx, imgs in tqdm(enumerate(dataloader), leave = False, total = len(dataloader)):
                imgs = imgs.cuda(1)
                out = feature_extractor(imgs)
                list_of_features.extend(out.cpu().detach().numpy().tolist())
                if len(list_of_features) >= 5000:
                    gset = np.array(list_of_features)
                    idxs = self.selection(gset, sub_selection_technique, percentage)
                    final_idxs.extend(idxs)
                    list_of_features = []
        return final_idxs