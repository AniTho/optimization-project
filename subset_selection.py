import argparse
from torchvision import transforms
from dataload.mnist_load import MNISTDataset
from dataload.celeba_load import CelebaDataset
from dataload.lsun_load import LsunDataset
from dataload.cifar_load import CIFARDataset

def range_checker(min_value,max_value):
    def float_range_checker(arg):
        try:
            value = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if value < min_value or value > max_value:
            raise argparse.ArgumentTypeError(f'Value must be in range [{min_value}, {max_value}]')
        return value
    return float_range_checker

def main(args):
    print(args.dataset)
    transform = transforms.Compose([transforms.Resize(64),
                                transforms.RandomCrop((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
    if args.dataset == 'mnist':
        if args.subset:
            MNISTDataset(subset=args.subset, sub_selection_technique=args.technique, 
                         percentage=args.percentage, transform=transform)
    elif args.dataset == 'celeba':
        if args.subset:
            CelebaDataset(subset=args.subset, sub_selection_technique=args.technique, 
                         percentage=args.percentage, transform=transform)
            
    elif args.dataset == 'cifar10':
        if args.subset:
            CIFARDataset(subset=args.subset, sub_selection_technique=args.technique, 
                         percentage=args.percentage, transform=transform)
            
    else:
        if args.subset:
            LsunDataset(subset=args.subset, sub_selection_technique=args.technique, 
                         percentage=args.percentage, transform=transform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''It generates a txt file containing indices of elements for Data Subset selection
which is used for training since it is non adaptive data subset selection.''')
    parser.add_argument('-s', '--subset', action="store_true", help='If flag present then subset selection will be used')
    parser.add_argument('--technique', action='store', default='facility_location', 
                        type=str, help = 'Choose the subset selection technique from the given list',
                        choices=['facility_location', 'random', 'disparity_min', 'disparity_sum', 'log_determinant'])
    parser.add_argument('--percentage', action='store', type=range_checker(0, 1), default=0.1,
                        help = 'Percentage of points that has to be kept')
    parser.add_argument('-d','--dataset', action='store', type=str, default='mnist',
                        help='Select the dataset to run experiment on',
                        choices=['mnist', 'lsun', 'celeba', 'cifar10'])
    args = parser.parse_args()
    main(args)