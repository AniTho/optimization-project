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

class MNISTDataset(Dataset):
    def __init__(self, subset = False, sub_selection_technique = None,  percentage = 0.1, 
                 transform = None, bs = 64):
        mnist = datasets.MNIST(root = 'data/', train = True, download=True)
        self.data = mnist.data
        if not subset:
            self.idxs = list(range(len(self.data)))
        else:
            self.idxs = self.subset_selection(sub_selection_technique, percentage, bs)
            np.savetxt(f'mnist_{sub_selection_technique}.txt', np.array(self.idxs))
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

    def subset_selection(self, sub_selection_technique, percentage, bs):
        budget = int(percentage*len(self.data))
        print(f'{"*"*20} Selecting Subset {"*"*20}')
        if sub_selection_technique == 'random':
            all_idxs = list(range(len(self.data)))
            idxs = random.sample(all_idxs, budget)
            return idxs
        print(f'{"#"*10} Preparing Ground Set {"#"*10}')
        gset = self.prepare_ground_set(bs)
        n_ground = len(gset)
        print(f'{"*"*10} Running Subset Selection {"*"*10}')
        final_idxs = []
        per_batch_samples = 5000
        num_iteration = n_ground//per_batch_samples
        for curr_idx in tqdm(range(num_iteration), leave = False, total = num_iteration):
            gset_sub = gset[curr_idx*per_batch_samples:(curr_idx+1)*per_batch_samples].copy()
            budget = int(percentage*len(gset_sub))
            if sub_selection_technique == 'facility_location':
                sub_selection_method = FacilityLocationFunction(n = len(gset_sub), mode = 'dense', data = gset_sub, metric='euclidean')
            elif sub_selection_technique == 'disparity_min':
                sub_selection_method = DisparityMinFunction(n = len(gset_sub), mode = 'dense', data = gset_sub, metric='euclidean')
            elif sub_selection_technique == 'disparity_sum':
                sub_selection_method = DisparitySumFunction(n = len(gset_sub), mode = 'dense', data = gset_sub, metric='euclidean')
            elif sub_selection_technique == 'log_determinant':
                sub_selection_method = LogDeterminantFunction(n = len(gset_sub), mode = 'dense', data = gset_sub, metric='euclidean')
            else:
                raise Exception("Mentioned method for subset selection is not available")
            
            list_of_idxs = sub_selection_method.maximize(budget, optimizer='StochasticGreedy', show_progress = False)
            idxs = [x[0] for x in list_of_idxs]
            final_idxs.extend(idxs)
        return final_idxs
    
    def prepare_ground_set(self, bs):
        # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.RandomCrop(64, 64),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])
                                        ])
        feature_extractor = FeatureExtractor('resnet50')
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
        with torch.no_grad():
            print('Forming Batches for generating ground set')
            complete_dataset = MNISTDataset(transform = transform)
            dataloader = DataLoader(complete_dataset, batch_size = bs, shuffle=False)
            gset = np.zeros((len(complete_dataset), feature_extractor.num_features), dtype = np.float32)
            for idx, imgs in tqdm(enumerate(dataloader), leave = False, total = len(dataloader)):
                imgs = imgs.to(device)
                out = feature_extractor(imgs)
                gset[idx*bs:(idx+1)*bs, :] = out.cpu().detach().numpy()
        return gset