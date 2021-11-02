import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train on')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--path', type=str, default='data', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--noise_mode', type=str, default='dependent', help='Noise mode')
parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
parser.add_argument('--sigma', type=float, default=0.5, help='STD of Gaussian noise')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
args = parser.parse_args()
