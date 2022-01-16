import argparse
from train_sound_SLN import run

"""
Script to run the ablation study on the CIFAR data to reproduce the results in the paper.

"""

parser = argparse.ArgumentParser()
parser.description("This script is used to run abliation study on CIFAR-10")
parser.add_argument('--runs', type=int, default=5, help='number of runs')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--sigma', type=float, default=0.5, help='standard deviation of noise')


def run_experiment(args):
    print("Running experiment with the following parameters: ")
    print(args)
    for i in range(args.runs):
        print(f"Run {i}")
        run(sigma=args.sigma, epochs=args.epochs, experiment=True)

def plot_experiment(args):
    """
        Plotting the experiments
    """
    pass


def main():
    args = parser.parse_args()
    run_experiment(args)
    plot_experiment(args)
if __name__ == '__main__':
    main()