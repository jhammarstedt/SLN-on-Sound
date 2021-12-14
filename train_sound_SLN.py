import argparse
import json
import os
import psutil
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from fsd50_src.src.data.transforms import get_transforms_fsd_chunks
from fsd50_src.src.data.utils import _collate_fn_multiclass, _collate_fn
from fsd50_src.src.utilities.config_parser import parse_config, get_data_info
from fsd50_src.src.data.dataset import SpectrogramDataset
from fsd50_src.src.models.fsd50k_lightning import model_helper
from fsd50_src.src.data.fsd_eval_dataset import FSD50kEvalDataset,_collate_fn_eval


parser = argparse.ArgumentParser()
parser.description = "Training script for FSD50k baselines"
parser.add_argument("--cfg_file", type=str, help='path to cfg file')
parser.add_argument("--expdir", "-e", type=str, help="directory for logging and checkpointing")
parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cw", type=str, required=False,
                    help="path to serialized torch tensor containing class weights")
parser.add_argument("--resume_from", type=str,
                    help="checkpoint path to continue training from")
parser.add_argument('--mixer_prob', type=float, default=0.75,
                    help="background noise augmentation probability")
parser.add_argument("--fp16", action="store_true",
                    help='flag to train in FP16 mode')
parser.add_argument("--gpus", type=int, default=0,
                    help="Single or multiple gpus to train on. For multiple, use ', ' delimiter ")

parser.add_argument("--runs", type=int, default=5,
                    help="...")

parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--stdev", type=float, default=0.5,
                    help="How much added noise")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="...")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="...")
parser.add_argument("--batch_size", type=int, default=256,
                    help="...")
parser.add_argument("--correction", type=int, default=20,
                    help="...")
parser.add_argument("--num_class", type=int, default=1,
                    help="...")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="...")
parser.add_argument("--sigma", type=int, default=1,
                    help=" for cifar 10: 1 == symmetric, 0.5 == asymetric")
parser.add_argument("--pretrained", type=bool, default=False,
                    help=" Set to true to use pretrained model")

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

    def step(self):  # update moving average
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if param.type() == 'torch.cuda.LongTensor':
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
        for data, _, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target, reduction='none')

            output = F.softmax(output, dim=1)

            losses.append(loss.cpu().numpy())
            softmax.append(output.cpu().numpy())
    #print(len(np.concatenate(softmax)))
    #print(len(np.concatenate(losses)))
    return np.concatenate(softmax), np.concatenate(losses)


# Train on ResNet-28-2
def train(args, model, device, loader, optimizer, epoch, ema_optimizer, criterion):
    model.train()
    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)

    # data, complex_targets, target
    i = 0
    for data, _, target in tqdm(loader):
        #if i == 10:
        #    break
        #else:
        #    i+=1
        # One-hot encode single-digit labels
        # if len(target.size()) == 1:
        #     target = torch.zeros(target.size(0), args.num_class.scatter_(1, target.view(-1, 1), 1))
        # print(target.shape)
        if len(target.size()) == 1:
            one_hot_target = torch.tensor(np.eye(args.num_class)[target])
        else:
            one_hot_target = target

        data, target = data.to(device), one_hot_target.to(device)

        # SLN
        #if args.sigma > 0:
        #    target += args.sigma * torch.randn(target.size(), device=device)

        # Calculate loss
        output = model(data)
        loss = criterion(output, target)

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
    i = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            #if i == 10:
            #    break
            #else:
            #    i += 1
            #one_hot_target = torch.tensor(np.eye(args.num_class)[target])

            data, target = data.to(device), target.to(device)

            # Calculate cross-entropy loss
            output = model(data)
            output = output.mean(0).unsqueeze(0)
            test_loss += criterion(output, target)

            # Check for correct prediction
            pred = output.argmax(dim=1, keepdim=True)

            if len(target.size()) == 2:
                target = target.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum()

    # Return average test loss and test accuracy
    return test_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def save_log(log, train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA):
    log['train_loss'].append(train_loss)
    log['train_acc'].append(train_acc)
    log['test_loss'].append(test_loss)
    log['test_acc'].append(test_acc)
    print(f'Train acc: {train_acc}, Test acc: {test_acc}, Test acc no EMA: {test_acc_NoEMA}')
    log['test_loss_NoEMA'].append(test_loss_NoEMA)
    log['test_acc_NoEMA'].append(test_acc_NoEMA)
    return log


def train_dataloader(train_set, args, collate_fn, shuffle):
    return DataLoader(train_set, num_workers=args.num_workers, shuffle=shuffle,
                      sampler=None, collate_fn=collate_fn,
                      batch_size=args.cfg['opt']['batch_size'],
                      pin_memory=False, drop_last=False)


def val_dataloader(val_set, args, shuffle):
    return DataLoader(val_set, sampler=None, num_workers=args.num_workers,
                      collate_fn=_collate_fn_eval,
                      shuffle=shuffle, batch_size=1,
                      pin_memory=False)

def load_data(args):
    """
    Reads the data from the specified directory in the args and loads the data into the specified dataloader

    Args:
        args ([type]): [description]

    Returns:
        train_loader,test_loader,train_eval_loader
    """
    if args.cfg['model']['type'] == "multiclass":
        collate_fn = _collate_fn_multiclass
        mode = "multiclass"
    elif args.cfg['model']['type'] == "multilabel":
        collate_fn = _collate_fn
        mode = "multilabel"
    else:
        raise ValueError("Model type not supported")

    trainset = SpectrogramDataset(args.cfg['data']['train'],
                                  args.cfg['data']['labels'],
                                  args.cfg['audio_config'],
                                  mode=mode, augment=True,
                                  mixer=args.tr_mixer,
                                  transform=args.tr_tfs)
    testset = FSD50kEvalDataset(args.cfg['data']['val'],
                                args.cfg['data']['labels'],
                                args.cfg['audio_config'],
                                transform=args.val_tfs
                                )

    train_loader = train_dataloader(trainset, args=args, collate_fn=collate_fn, shuffle=True)
    test_loader = val_dataloader(testset, args=args, shuffle=False)
    train_eval_loader = train_dataloader(trainset, collate_fn=collate_fn,args=args, shuffle=False)

    return train_loader, test_loader, train_eval_loader, trainset, testset

def save(model, ema_model, log):
    try:
        torch.save(model.state_dict(), 'model_state')
        print('Successfully saved model params')
    except:
        print('Failed to save model params')

    try:
        torch.save(ema_model.state_dict(), 'ema_model_state')
        print('Successfully saved ema_model params')
    except:
        print('Failed to save ema_model params')

    try:
        with open('training_log.json', 'w') as f:
            json.dump(log, f)
        print('Successfully saved training log')
    except:
        print('Failed to save training log')


def run(args):
    process = psutil.Process(os.getpid())
    print(f'Initial memory usage: {process.memory_info().rss * 9.31e-10}')

    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training')
    torch.cuda.set_device(args.gpu_id)

    # load data
    train_loader, test_loader, train_eval_loader, trainset, testset = load_data(args)

    args.num_class = args.cfg['model']['num_classes']
    print(f'Number of classes: {args.num_class}')

    noisy_targets = trainset.labels
    noisy_targets = np.eye(args.num_class)[noisy_targets]

    args.cfg['model']['pretrained'] = args.pretrained
    model = model_helper(args.cfg['model']).cuda()
    # MO model
    # ema_model = Wide_ResNet(num_classes=args.num_class).cuda()
    args.cfg['model']['pretrained_path'] = args.cfg['model']['pretrained_path_ema']
    ema_model = model_helper(args.cfg['model']).cuda()
    for param in ema_model.parameters():
        param.detach_()

    # Optimizers
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    ema_optimizer = WeightEMA(model, ema_model)

    # criterion = BCEWithLogitsLoss(args.cw)
    criterion = F.cross_entropy

    log = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_loss_NoEMA': [],
        'test_acc_NoEMA': []
    }

    print(f'Memory usage before training: {process.memory_info().rss * 9.31e-10}')

    # Training loop
    total_t0 = time.time()
    print('Starting training loop ...')
    for epoch in range(21, args.epochs + 1):
        t0 = time.time()

        # Label Correction on 250th epoch, without tuning
        if epoch > args.correction:
            if args.cfg['model']['type'] == "multiclass":
                collate_fn = _collate_fn_multiclass
                mode = "multiclass"
            elif args.cfg['model']['type'] == "multilabel":
                collate_fn = _collate_fn
                mode = "multilabel"
            else:
                raise ValueError("Model type not supported")
            
            args.sigma = 0  # Stop SLN

            output, losses = get_output(ema_model, device, train_eval_loader, args)
            output = np.eye(args.num_class)[output.argmax(axis=1)]  # Predictions

            losses = (losses - min(losses)) / (max(losses) - min(losses))  # Normalize to range [0, 1]
            losses = losses.reshape([len(losses), 1])

            targets = losses * noisy_targets + (1 - losses) * output  # Label correction
            trainset.labels = np.argmax(targets, axis=1)
            train_loader = train_dataloader(trainset, args=args, collate_fn=collate_fn, shuffle=True)

        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, ema_optimizer, criterion)
        test_loss, test_acc = test(args, ema_model, device, test_loader, criterion)
        test_loss_NoEMA, test_acc_NoEMA = test(args, model, device, test_loader, criterion)
        log = save_log(log, train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA)
        print('\nEpoch: {} Time: {:.1f}s.'.format(epoch, time.time() - t0))
        print('Train loss:\t{:.3f}\tTest loss:\t{:.3f}\tTest loss NoEMA:\t{:.3f}\t'.format(train_loss, test_loss,
                                                                                           test_loss_NoEMA))
        save(model, ema_model, log)

    print('\nTotal training time: {:.1f}s.\n'.format(time.time() - total_t0))

    save(model, ema_model, log)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    args.output_directory = os.path.join(args.expdir, "ckpts")
    args.log_directory = os.path.join(args.expdir, "logs")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    cfg = parse_config(args.cfg_file)
    data_cfg = get_data_info(cfg['data'])
    cfg['data'] = data_cfg
    args.cfg = cfg

    args.tr_mixer = None

    tr_tfs = get_transforms_fsd_chunks(True, 101)
    val_tfs = get_transforms_fsd_chunks(False, 101)

    args.tr_tfs = tr_tfs
    args.val_tfs = val_tfs

    run(args)
