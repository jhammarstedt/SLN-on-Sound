import time

import torch
import numpy as np
import logging as log
import torch.nn.functional as F

from args import get_args
from data import get_cifar
from network import Wide_ResNet
from helpers import WeightExponentialMovingAverage, TrainingLogger
#from ablation_study.abliation_study import run_experiment

import json
import matplotlib.pyplot as plt
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = "cpu"

def get_data_loader(data, batch_size, num_workers=4, shuffle=False):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_data(args):
    # TODO make sure all labels are one-hot encoded
    log.info(f'Loading {args.dataset} dataset ...')

    train_set, test_set = get_cifar(dataset=args.dataset)
    # one-hot
    train_set.targets = np.eye(args.num_class)[train_set.targets]

    train_loader = get_data_loader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = get_data_loader(test_set, args.batch_size, num_workers=args.num_workers)
    train_eval_loader = get_data_loader(train_set, args.batch_size, num_workers=args.num_workers)

    assert args.num_class == len(train_set.classes)
    return train_loader, train_set, test_loader, train_eval_loader


def get_models(args):
    model = Wide_ResNet(num_classes=args.num_class)
    momentum_model = Wide_ResNet(num_classes=args.num_class)

    for param in momentum_model.parameters():
        param.detach_()
    return model.to(DEVICE), momentum_model.to(DEVICE)


def get_prediction(model, data_loader):
    log.debug('Getting predictions ...')
    model.eval()

    predictions = []
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            Y_pred = model(X)
            output = F.softmax(Y_pred, dim=1)
            loss = -torch.sum(torch.log(output) * Y, dim=1)

            losses.append(loss.cpu().numpy())
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(losses)


def label_correction(args, momentum_model, train_eval_loader, train_set, original_labels):
    log.debug('Correcting labels ...')
    predictions, losses = get_prediction(momentum_model, train_eval_loader)
    predictions_one_hot = np.eye(args.num_class)[predictions.argmax(axis=1)]  # Predictions

    min_loss, max_loss = losses.min(), losses.max()

    normalized_loss = (losses - min_loss) / (max_loss - min_loss)
    normalized_loss = normalized_loss[:, None]

    # Label correction
    y_correction = normalized_loss * original_labels + (1 - normalized_loss) * predictions_one_hot

    # update labels
    train_set.targets = y_correction
    return get_data_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)

def start_training(
    args,momentum_model,
    model,train_loader,test_loader,
    train_eval_loader,optimizer,
    optimizer_momentum, training_log,
    train_set,original_train_Y):


    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if epoch >= args.correction:
            args.sigma = 0.
            label_correction(args, momentum_model, train_eval_loader, train_set, original_train_Y)

        train_loss, train_acc = _train_step(args, model, train_loader, optimizer, optimizer_momentum)
        test_loss, test_acc = _test_step(momentum_model, test_loader)
        test_loss_NoEMA, test_acc_NoEMA = _test_step(model, test_loader)
        training_log.save_epoch(train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA)
        training_log.print_last_epoch(epoch=epoch, logger=log, time=time.time() - epoch_start)

    if args.abliation_study:
        return training_log


def main(args):
    log.getLogger().setLevel(args.loglevel.upper())
    log.info(f'Using {DEVICE} for torch training.')

    train_loader, train_set, test_loader, train_eval_loader = get_data(args)
    # original_train_Y = np.eye(args.num_class)[train_set.targets]
    original_train_Y = train_set.targets.copy()


    model, momentum_model = get_models(args)
    momentum_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer_momentum = WeightExponentialMovingAverage(model, momentum_model)
    training_log = TrainingLogger()

    print("####### Starting training #######")
    
    training_bag = [
        momentum_model,model,
        train_loader,test_loader,train_eval_loader,
        optimizer,optimizer_momentum, 
        training_log,
        train_set,original_train_Y
        ]
    if not args.abliation:
        start_training(args,*training_bag)
    else: # if abliation study
        abs_study = Abliation_study(parameter="sigma")
        abs_study.run_experiment(args, *training_bag)
        abs_study.plot_experiment(args)

def _train_step(args, model, data_loader, optimizer, momenturm_optimizer):
    log.debug('Train step ...')
    batch_losses = []
    batch_predictions = []
    model.train()
    for _X, _y in data_loader:
        optimizer.zero_grad()

        X, y = _X.to(DEVICE), _y.to(DEVICE)

        # add stochastic label noise
        y += args.sigma * torch.randn(y.size(), device=DEVICE)

        y_pred = model(X)
        loss_per_input = -torch.sum(F.log_softmax(y_pred, dim=1) * y, dim=1)
        batch_loss = -torch.mean(loss_per_input)

        # backpropagation
        batch_loss.backward()
        optimizer.step()
        momenturm_optimizer.step()

        batch_predictions += y.argmax(dim=1).eq(y_pred.argmax(dim=1)).tolist()
        batch_losses += loss_per_input.detach().tolist()

    return sum(batch_losses) / len(batch_losses), sum(batch_predictions) / len(batch_predictions)


def _test_step(model, data_loader):
    log.debug('Test step ...')
    batch_losses = []
    batch_predictions = []
    model.eval()
    with torch.no_grad():
        for _X, _y in data_loader:
            X, y = _X.to(DEVICE), _y.to(DEVICE)
            y_pred = model(X)
            batch_losses += F.cross_entropy(y_pred, y, reduction='none').tolist()
            batch_predictions += y.eq(y_pred.argmax(dim=1)).tolist()

    return sum(batch_losses) / len(batch_losses), sum(batch_predictions) / len(batch_predictions)

class Abliation_study:

    """
    This class is used to run abliation study.
    """
    def __init__(self,args,parameter="sigma") -> None:
        
        if parameter =="sigma":
            self.sigmas = [0,0.2,0.4,0.6,0.8,1] #sigmas to test on
        else:
            raise NotImplementedError("Only sigma is implemented")
        
        self.types  = ["symmetric","asymmetric"] # we have implementation for these two types

        self.logs = {
            t:{f"sigma_{s}":{f"run_{i}":{} for i in range(
                args.runs)} for s in self.sigmas} for t in self.types}#ex {symmetric: {sigma_0.2: {run_3: {"training":[],"test":[]}}}}

    def run_experiment(self,args,training_bag):
        print("Running ablation study with {} runs and {} epochs".format(args.runs,args.epochs))
        #print(args)
        if args.parameter == "sigma":
            
            #logs = {t:{f"sigma_{s}":{f"run_{i}":{} for i in range(args.runs)} for s in sigmas} for t in types}#ex {symmetric: {sigma_0.2: {run_3: {"training":[],"test":[]}}}}
            
            for t in self.types:
                for sigma in self.sigmas:
                    for i in range(args.runs):
                        #run_result = run(sigma=sigma, epochs=args.epochs, experiment=True, type=t) 
                        training_log = start_training(args,*training_bag)
                        
                        data = {
                            'train_loss': training_log.train_loss,
                            'train_acc': training_log.train_acc,
                            'test_loss': training_log.test_loss,
                            'test_acc': training_log.test_acc,
                            'test_loss_NoEMA': training_log.test_loss_NoEMA,
                            'test_acc_NoEMA': training_log.test_acc_NoEMA
                        }

                        self.log_results(t,i,sigma,data)
 


            #self.plot_experiment(self,args)
        else:
            raise NotImplementedError("Only sigma is implemented for now")

    def log_results(self,t,i,sigma,data):
        self.logs[t][f"sigma_{sigma}"][f"run_{i}"].update(data)
        with open('ablation_study/results/training_log.json', 'w') as f: #saving results after each run just in case
            json.dump(self.logs, f)

    def plot_experiment(self,args):
        #types = ["symmetric","asymmetric"] # we have implementation for these two types
        #sigmas = [0,0.2,0.4,0.6,0.8,1] #sigmas to test on

        compiled_results = {t:[(int,float) for s in range(len(self.sigmas))] for t in self.types} # ex {symmetric: [(0,0.2),(1,0.4),(2,0.6),(3,0.8),(4,1)]}



        with open('ablation_study/results/training_log.json', 'r') as f:
            res = json.load(f)
        for t in self.types:
            for sigma_index,s in enumerate(self.sigmas):
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
    






if __name__ == '__main__':
    main(get_args())
