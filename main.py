import argparse
from train import train
from dataload.celeba_load import CelebaDataset
from dataload.mnist_load import MNISTDataset
from dataload.lsun_load import LsunDataset
from dataload.cifar_load import CIFARDataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from models.discriminator import DCGanDiscriminator
from models.generator import DCGanGenerator
from subset_selection import range_checker
from utils.utils import determinantal_point_process
import wandb

def main(args):
    transform = transforms.Compose([transforms.Resize(64),
                                transforms.RandomCrop((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], 
                                                     [0.5, 0.5, 0.5])])
    
    # Hyperparameters
    latent_dim = args.latent
    num_epochs = args.num_epochs
    lr = 0.0002
    beta_1 = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size

    # wandb setup
    if args.dpp and args.subset:
        wandb.init(dir = 'runs/', project='Optimization Project', 
                   name = f"full_with_div_{args.weight} ({args.dataset})_{args.percentage}_{args.technique}")
    elif args.dpp and not(args.subset):
        wandb.init(dir = 'runs/', project='Optimization Project', name = f'dcgan_full_with_div_{args.weight} ({args.dataset})')
    elif not(args.dpp) and args.subset:
        wandb.init(dir = 'runs/', project='Optimization Project', 
                   name = f'dcgan_full_without_div ({args.dataset}_{args.percentage}_{args.technique}')
    else:
        wandb.init(dir = 'runs/', project='Optimization Project', name = f'dcgan_full_without_div ({args.dataset})')
    wandb.config = {
    'batch_size': batch_size,
    'latent_dim': latent_dim,
    'image_size': batch_size,
    'beta1 (for adam)': beta_1,
    'learning_rate': lr,
    'num_epochs': num_epochs,
    'dataset': args.dataset,
    'subset_percent': args.percentage,
    'DPP Lambda': args.weight,
    'loss': 'bce_logits'
    }

    criterion = torch.nn.BCEWithLogitsLoss()
    if args.dataset == 'mnist':
        dataset = MNISTDataset(args.subset, args.technique, args.percentage, transform=transform)
    elif args.dataset == 'celeba':
        dataset = CelebaDataset(args.subset, args.technique, args.percentage, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = CIFARDataset(args.subset, args.technique, args.percentage, transform=transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=8)
    generator, discriminator = DCGanGenerator(latent_dim=latent_dim), DCGanDiscriminator()
    gen_opt = torch.optim.AdamW(generator.parameters(), lr = lr, betas = (beta_1, 0.999))
    disc_opt = torch.optim.AdamW(discriminator.parameters(), lr = lr, betas = (beta_1, 0.999))
    generator_losses, discriminator_losses, diversities = train(dataloader, generator, discriminator, num_epochs, criterion,
                                                                gen_opt, disc_opt, latent_dim, None, determinantal_point_process,
                                                                div_flag = args.dpp, div_lambda = args.weight, device = device)
    print(generator_losses.mean())
    print(discriminator_losses.mean())
    print(diversities.mean())
    wandb.finish(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Start Training of GAN''')
    parser.add_argument('-s', '--subset', action="store_true", help='If flag present then subset selection will be used')
    parser.add_argument('-t', '--technique', action='store', default='facility_location', 
                        type=str, help = 'Choose the subset selection technique from the given list',
                        choices=['facility_location', 'random', 'disparity_min', 'disparity_sum', 'log_determinant'])
    parser.add_argument('--percentage', action='store', type=range_checker(0, 1), default=0.1,
                        help = 'Percentage of points that has to be kept')
    parser.add_argument('-d','--dataset', action='store', type=str, default='mnist',
                        help='Select the dataset to run experiment on',
                        choices=['mnist', 'celeba', 'cifar10'])
    parser.add_argument('-l', '--latent', action='store', type = str, default = 100,
                        help='Set the latent dimension from which to sample the data')
    parser.add_argument('-e', '--num_epochs', action='store', type=int, default = 30,
                        help = 'Set the number of epochs')
    parser.add_argument('-b', '--batch_size', action = 'store', type=int, default = 64,
                        help = 'Set the batch size for training')
    parser.add_argument('-dpp', '--dpp', action='store_true', help='If flag present then diversity detection will be used')
    parser.add_argument('-w', '--weight', action='store', type=float, default=0.1,
                        help='Hyperparameter to put weightage on the dpp')
    args = parser.parse_args()
    main(args)