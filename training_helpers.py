import torch
import numpy as np
import logging as log
import torch.nn.functional as F
import json
from data import get_cifar
from network import Wide_ResNet

from fsd50_src.src.data.utils import _collate_fn_multiclass, _collate_fn
from fsd50_src.src.models.fsd50k_lightning import model_helper
from fsd50_src.src.data.dataset import SpectrogramDataset
from fsd50_src.src.data.fsd_eval_dataset import FSD50kEvalDataset, _collate_fn_eval

# Keep track of loss and acc without moving data to cpu for faster training
class BatchRecord:
    def __init__(self, device, dataloader):
        self.batch_losses = torch.zeros(1, device=device)
        self.batch_predictions = torch.zeros(1, device=device)
        self.len = len(dataloader.dataset)

    def update(self, X, y, y_pred, loss):
        self.batch_losses += loss
        if len(y.size())==2: y = y.argmax(dim=1, keepdim=True) # For testing step
        self.batch_predictions += y.eq(y_pred.argmax(dim=1)).sum()

    def get_record(self):
        return self.batch_losses.item() / self.len, self.batch_predictions.item() / self.len

def SLN(y, device, sigma):
    """
    Add stochastic label noise

    params:
        y: (batch_size, num_class)
        device: cuda or cpu
        sigma: float
    """
    label_noise = torch.randn(y.size(), device=device)
    return y + sigma * label_noise

def get_data_loader_FSD(data, batch_size, collate_fn, num_workers=4, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)

def get_data_loader(data, batch_size, num_workers=4, shuffle=False):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def get_cifar_data(args):
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

def get_FSD_data(args):
    """
    Data loader from FSD50k dataset, with preprocessing
    
    """

    collate_fn = _collate_fn_multiclass
    mode = "multiclass"

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

    train_loader = get_data_loader_FSD(trainset, collate_fn=collate_fn, shuffle=True, batch_size=args.cfg['opt']['batch_size'])
    test_loader = get_data_loader_FSD(testset, collate_fn=_collate_fn_eval, shuffle=False, batch_size=1, drop_last=True)
    train_eval_loader = get_data_loader_FSD(trainset, collate_fn=collate_fn, shuffle=False, batch_size=args.cfg['opt']['batch_size'])

    return train_loader, trainset, test_loader, train_eval_loader

def get_models(args, device):
    model = Wide_ResNet(num_classes=args.num_class)
    momentum_model = Wide_ResNet(num_classes=args.num_class)

    for param in momentum_model.parameters():
        param.detach_()

    return model.to(device), momentum_model.to(device)

def get_FSD_models(args):
    model = model_helper(args.cfg['model']).cuda()
    momentum_model = model_helper(args.cfg(['model'])).cuda()
    momentum_model.load_state_dict(model.state_dict())

    return model, momentum_model

def label_correction(args,epoch, momentum_model, train_eval_loader, train_set, original_labels, device):
    """
    Here we will run the lbl correction if the epoch over the threshold
    """
    
    if epoch < args.correction: # No correction
        return get_data_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    log.debug('Correcting labels ...')
    args.sigma =0.
    momentum_model.eval()

    predictions = []
    losses = []

    with torch.no_grad():
        for X, Y in train_eval_loader:
            X, Y = X.to(device), Y.to(device)
            Y_pred = momentum_model(X)
            output = F.softmax(Y_pred, dim=1)
            loss = -torch.sum(torch.log(output) * Y, dim=1)

            losses.append(loss.cpu().numpy())
            predictions.append(output.cpu().numpy())

    predictions, losses = np.concatenate(predictions), np.concatenate(losses)
    predictions_one_hot = np.eye(args.num_class)[predictions.argmax(axis=1)]  # Predictions

    min_loss, max_loss = losses.min(), losses.max()

    normalized_loss = (losses - min_loss) / (max_loss - min_loss)
    normalized_loss = normalized_loss[:, None]

    # Label correction
    y_correction = normalized_loss * original_labels + (1 - normalized_loss) * predictions_one_hot

    # update labels
    train_set.targets = y_correction
    return get_data_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)

class WeightExponentialMovingAverage:
    def __init__(self, model, momentum_model, alpha=0.999):
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.momentum_params = list(momentum_model.state_dict().values())

    def step(self):
        for param, momentum_param in zip(self.params, self.momentum_params):
            momentum_param.copy_(momentum_param.data*self.alpha + param.data*(1.0 - self.alpha))


class TrainingLogger:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.test_loss_NoEMA = []
        self.test_acc_NoEMA = []

    def save_epoch(self, train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)
        self.test_loss_NoEMA.append(test_loss_NoEMA)
        self.test_acc_NoEMA.append(test_acc_NoEMA)

    def print_last_epoch(self, epoch, logger=None, time=None):
        time_info = ''
        if time:
            time_info = '\tTime: {:.1f} min'.format(time / 60)
        log_string = 'Epoch {} {}\tTrain loss:\t{:.3f}\tTest loss:\t{:.3f}\tTest loss NoEMA:\t{:.3f}\t'.format(
            epoch, time_info, self.train_loss[-1], self.test_loss[-1], self.test_loss_NoEMA[-1])
        if not logger:
            print(log_string)
        else:
            logger.info(log_string)

    def export_as_json(self, path):
        data = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'test_loss': self.test_loss,
            'test_acc': self.test_acc,
            'test_loss_NoEMA': self.test_loss_NoEMA,
            'test_acc_NoEMA': self.test_acc_NoEMA
        }
        with open(path, 'w') as f:
            json.dump(data, f)
