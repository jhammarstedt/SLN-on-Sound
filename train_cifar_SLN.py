from sys import platform
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data import get_cifar
from resnet import Wide_ResNet

# Weight Exponential Moving Average
class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        # copy params
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self): # update moving average
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if param.type()=='torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)

# Get output
def get_output(model, device, loader):
    softmax = []
    losses = []
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if len(target.size()) == 1:
                loss = F.cross_entropy(output, target, reduction="none")
            else:
                loss = -torch.sum(F.log_softmax(output, dim=1) * target, dim=1)

            output = F.softmax(output, dim=1)

            losses.append(loss.cpu().numpy())
            softmax.append(output.cpu().numpy())

    return np.concatenate(softmax), np.concatenate(losses)

# Train on Wide ResNet-28-2
def train(args, model, device, loader, optimizer, epoch, ema_optimizer):
    model.train()
    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)

    for data, target in loader:
        # One-hot encode single-digit labels
        if len(target.size()) == 1:
            target = torch.zeros(target.size(0), args['num_class']).scatter_(1, target.view(-1,1), 1)

        data, target = data.to(device), target.to(device)

        # SLN
        if args['sigma'] > 0:
            target += args['sigma'] * torch.randn(target.size(), device=device)

        # Calculate loss
        output = model(data)
        loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update MO model
        if ema_optimizer:
            ema_optimizer.step()

        # Accumulate batch loss
        train_loss += data.size(0) * loss
        # Predicted class
        prediction = output.argmax(dim=1, keepdim=True)

        if len(target.size()) == 2:
            target = target.argmax(dim=1, keepdim=True)

        # Accumulate correct predictions
        correct += prediction.eq(target.view_as(prediction)).sum()

    # Return epoch-average loss and accuracy
    return train_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)

# Test loss and accuracy
def test(args, model, device, loader, criterion=F.cross_entropy):
    model.eval()
    test_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Calculate cross-entropy loss
            output = model(data)
            test_loss += criterion(output, target, reduction='sum')

            # Check for correct prediction
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

    # Return average test loss and test accuracy
    return test_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)

def run(workers=2):
    args = {
        'runs': 5,
        'epochs': 2,
        'stdev': 0.5,
        'lr': 0.001,
        'noise_rate': 0.4,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
        'correction': 1,
        'num_class': 1,

        'gpu_id': 0,

        # cifar10
        'sigma': 1 # symmetric
        #sigma: 0.5 # asymmetric

        # cifar100
        #sigma: 0.2
    }

    device = torch.device('cuda:'+str(args['gpu_id']) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args['gpu_id'])

    # Import datasets: Cifar-10, Cifar-100
    trainset, testset = get_cifar(dataset='cifar10')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=workers)
    train_eval_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=False, num_workers=workers)
    args['num_class'] = len(trainset.classes)

    noisy_targets = trainset.targets
    noisy_targets = np.eye(args['num_class'])[noisy_targets]

    # Wide ResNet28-2 model
    model = Wide_ResNet(num_classes=args['num_class']).cuda()

    # MO model
    ema_model = Wide_ResNet(num_classes=args['num_class']).cuda()
    for param in ema_model.parameters():
        param.detach_()

    # Optimizers
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    ema_optimizer = WeightEMA(model, ema_model)

    # Training loop
    total_t0 = time.time()
    for epoch in range(1, args['epochs']+1):
        t0 = time.time()

        # Label Correction on 250th epoch, without tuning
        if epoch > args['correction']:
            args['sigma'] = 0 # Stop SLN

            output, losses = get_output(ema_model, device, train_eval_loader)
            output = np.eye(args['num_class'])[output.argmax(axis=1)] # Predictions

            losses = (losses - min(losses)) / (max(losses) - min(losses)) # Normalize to range [0, 1]
            losses = losses.reshape([len(losses), 1])

            targets = losses * noisy_targets + (1 - losses) * output # Label correction
            trainset.targets = targets
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=1)

        _, train_acc = train(args, model, device, train_loader, optimizer, epoch, ema_optimizer)
        _, test_acc = test(args, ema_model, device, test_loader)
        _, test_acc_NoEMA = test(args, model, device, test_loader)
        print('\nEpoch: {} Time: {:.1f}s.\n'.format(epoch, time.time()-t0))


    print('\nTotal training time: {:.1f}s.\n'.format(time.time()-total_t0))


# Show results


if __name__ == '__main__':
    workers = 2
    if platform == 'win32':
        torch.multiprocessing.freeze_support()
        workers = 1
    run()
