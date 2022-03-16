import time
import os

import torch
import numpy as np
import logging as log
import torch.nn.functional as F
from tqdm import tqdm

from args import get_args
from data import get_cifar
from network import Wide_ResNet
from helpers import WeightExponentialMovingAverage, TrainingLogger

from fsd50_src.src.data.transforms import get_transforms_fsd_chunks
from fsd50_src.src.data.utils import _collate_fn_multiclass, _collate_fn
from fsd50_src.src.models.fsd50k_lightning import model_helper
from fsd50_src.src.data.dataset import SpectrogramDataset
from fsd50_src.src.utilities.config_parser import parse_config, get_data_info
from fsd50_src.src.data.fsd_eval_dataset import FSD50kEvalDataset, _collate_fn_eval

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
# torch.backends.cudnn.fastest = True
# torch.set_num_threads(1)

def get_data_loader(data, batch_size, collate_fn, num_workers=4, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)


def load_data(args):
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

    train_loader = get_data_loader(trainset, collate_fn=collate_fn, shuffle=True, batch_size=args.cfg['opt']['batch_size'])
    test_loader = get_data_loader(testset, collate_fn=_collate_fn_eval, shuffle=False, batch_size=1)
    train_eval_loader = get_data_loader(trainset, collate_fn=collate_fn, shuffle=False, batch_size=args.cfg['opt']['batch_size'])

    return train_loader, trainset, test_loader, train_eval_loader

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
        for X, _, Y in data_loader:
            # Y = torch.tensor(np.eye(args.num_class)[Y])
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
        log.export_as_json('training_log.json')
        print('Successfully saved training log')
    except:
        print('Failed to save training log')

def main(args):
    log.getLogger().setLevel(args.loglevel.upper())
    log.info(f'Using {DEVICE} for torch training.')

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

    args.cfg['model']['pretrained'] = args.pretrained

    train_loader, train_set, test_loader, train_eval_loader = load_data(args)
    # original_train_Y = np.eye(args.num_class)[train_set.targets]
    # original_train_Y = train_set.targets.copy()

    args.num_class = args.cfg['model']['num_classes']

    model = model_helper(args.cfg['model']).cuda()
    momentum_model = model_helper(args.cfg['model']).cuda()
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
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # if epoch >= args.correction:
        #     args.sigma = 0.
        #     label_correction(args, momentum_model, train_eval_loader, train_set, original_train_Y)

        train_loss, train_acc = _train_step(args, model, train_loader, optimizer, optimizer_momentum)
        test_loss, test_acc = _test_step(momentum_model, test_loader)
        test_loss_NoEMA, test_acc_NoEMA = _test_step(model, test_loader)
        training_log.save_epoch(train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA)
        training_log.print_last_epoch(epoch=epoch, logger=log, time=time.time() - epoch_start)

    save(model, momentum_model, training_log)

def _train_step(args, model, data_loader, optimizer, momenturm_optimizer):
    log.debug('Train step ...')
    model.train()
    batch_losses = torch.zeros(1, device=DEVICE)
    batch_predictions = torch.zeros(1, device=DEVICE)
    for _X, _, _y in tqdm(data_loader):
        optimizer.zero_grad()
        if len(_y.size()) == 1:
            one_hot = torch.tensor(np.eye(args.num_class)[_y], device=DEVICE)
        else:
            one_hot = _y

        X, y = _X.to(DEVICE), one_hot.to(DEVICE)

        # add stochastic label noise
        label_noise = torch.randn(y.size(), device=DEVICE)
        y += args.sigma * label_noise

        with torch.cuda.amp.autocast():
            y_pred = model(X)
            # batch_loss = F.cross_entropy(y_pred, y)
            loss_per_input = torch.sum(F.log_softmax(y_pred, dim=1) * y, dim=1).cuda()
            batch_loss = -torch.mean(loss_per_input).cuda()

        # backpropagation
        batch_loss.backward()
        optimizer.step()
        momenturm_optimizer.step()

        batch_predictions += y.argmax(dim=1).eq(y_pred.argmax(dim=1)).sum()
        batch_losses += X.size(0) * batch_loss

    return batch_losses.item() / len(data_loader.dataset), batch_predictions.item() / len(data_loader.dataset)


def _test_step(model, data_loader):
    log.debug('Test step ...')
    batch_losses = []
    batch_predictions = []
    model.eval()
    with torch.no_grad():
        for _X, _y in tqdm(data_loader):
            X, y = _X.to(DEVICE), _y.to(DEVICE)
            y_pred = model(X)
            batch_losses += F.cross_entropy(y_pred, y, reduction='none').tolist()
            batch_predictions += y.eq(y_pred.argmax(dim=1)).tolist()

    return sum(batch_losses) / len(batch_losses), sum(batch_predictions) / len(batch_predictions)


if __name__ == '__main__':
    main(get_args())
