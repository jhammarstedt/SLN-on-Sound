import argparse
from train_sound_SLN import run
import json
import matplotlib.pyplot as plt
import numpy as np
"""
Script to run the ablation study on the CIFAR data to reproduce the results in the paper.

"""

#TODO: ADD support to run the Run function with different parameters

parser = argparse.ArgumentParser()
parser.description("This script is used to run abliation study on CIFAR-10")
parser.add_argument('--runs', type=int, default=3, help='number of runs')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--parameter', type=str, default=None, 
help="What parameter to change in the ablation study: only sigma so far")

sigmas = [0,0.2,0.4,0.6,0.8,1] #sigmas to test on
types = ["symmetric","asymmetric"] # we have implementation for these two types

def run_experiment(args):
    print("Running experiment with the following parameters: ")
    print(args)
    if args.parameter == "sigma":
        
        logs = {t:{f"sigma_{s}":{f"run_{i}":{} for i in range(args.runs)} for s in sigmas} for t in types}#ex {symmetric: {sigma_0.2: {run_3: {"training":[],"test":[]}}}}
        logs = {t:{f"sigma_{s}":{f"run_{i}":{} for i in range(args.runs)} for s in sigmas} for t in types}#ex {symmetric: {sigma_0.2: {run_3: {"training":[],"test":[]}}}}
        
        for t in types:
            for sigma in sigmas:
                for i in range(args.runs):
                    run_result = run_result = run(sigma=sigma, epochs=args.epochs, experiment=True, type=t) 
                    logs[t][f"sigma_{sigma}"][f"run_{i}"].update(run_result)
                    with open('ablation_study/results/training_log.json', 'w') as f: #saving results after each run just in case
                        json.dump(logs, f)

    else:
        raise NotImplementedError("Only sigma is implemented for now")


def plot_experiment(args):

    compiled_results = {t:[(int,float) for s in range(len(sigmas))] for t in types} # ex {symmetric: [(0,0.2),(1,0.4),(2,0.6),(3,0.8),(4,1)]}



    with open('ablation_study/results/training_log.json', 'r') as f:
        res = json.load(f)
    for t in types:
        for sigma_index,s in enumerate(sigmas):
            avg = 0
            for r in range(args.runs):
                avg+= np.mean(res[t][f"sigma_{s}"][f"run_{r}"]["test_acc"][-1]) #get the last item
            
            compiled_results[t][sigma_index] = (s,avg/args.runs)
            
            
    # plot the result of the ablation study for each type, sigma
    print(compiled_results)
    for t in compiled_results:
        print(compiled_results[t])
        x,y = zip(*compiled_results[t])
        plt.scatter(x,y,label=t)
        plt.plot(x,y)
    plt.xlabel("Sigma")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.grid()
    plt.title("Performance of SLN w.r.t sigma")
    plt.show()
    plt.savefig("ablation_study/results/sigma_performance.png")
   






def main():
    args = parser.parse_args()
    run_experiment(args)
    plot_experiment(args)
if __name__ == '__main__':
    main()