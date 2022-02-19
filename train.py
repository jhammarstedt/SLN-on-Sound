import time

import torch
import numpy as np
import logging as log
import torch.nn.functional as F

from args import get_args
from data import get_cifar
from network import Wide_ResNet
from helpers import WeightExponentialMovingAverage, TrainingLogger


log.basicConfig(level=log.DEBUG)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    model.eval()

    predictions = []
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            Y_pred = model(X)

            # TODO when is the size == 1? do we need it?
            if len(Y.size()) == 1:
                loss = F.cross_entropy(Y_pred, Y, reduction="none")
            else:
                loss = -torch.sum(F.log_softmax(Y_pred, dim=1) * Y, dim=1)

            output = F.softmax(Y_pred, dim=1)

            losses.append(loss.cpu().numpy())
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(losses)


def label_correction(args, momentum_model, train_eval_loader, train_set, original_labels):
    predictions, losses = get_prediction(momentum_model, train_eval_loader)
    predictions_one_hot = np.eye(args.num_class)[predictions.argmax(axis=1)]  # Predictions

    min_loss, max_loss = losses.min(), losses.max()
    losses = (losses - min_loss) / (max_loss - min_loss)
    losses = losses.reshape([losses.shape[0], 1])

    # Label correction
    targets = losses * original_labels + (1 - losses) * predictions_one_hot

    # update labels
    train_set.targets = targets
    return get_data_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)


def main(args):
    log.info(f'Using {DEVICE} for torch training.')

    train_loader, train_set, test_loader, train_eval_loader = get_data(args)
    # original_train_Y = np.eye(args.num_class)[train_set.targets]
    original_train_Y = train_set.targets.copy()


    model, momentum_model = get_models(args)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer_momentum = WeightExponentialMovingAverage(model, momentum_model)
    training_log = TrainingLogger()


    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if epoch >= args.correction:
            args.sigma = 0.
            label_correction(args, momentum_model, DEVICE, train_eval_loader, train_set, original_train_Y)

        train_loss, train_acc = _train_step(args, model, train_loader, optimizer, optimizer_momentum)
        # test_loss, test_acc = _test_step(momentum_model, test_loader)
        # test_loss_NoEMA, test_acc_NoEMA = _test_step(model, test_loader)
        training_log.save_epoch(train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA)
        training_log.print_last_epoch(epoch=epoch, logger=log, time=time.time() - epoch_start)


def _train_step(args, model, data_loader, optimizer, momenturm_optimizer):
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
        # momenturm_optimizer.step()

        batch_predictions += y.argmax(dim=1).eq(y_pred.argmax(dim=1)).tolist()
        batch_losses += loss_per_input.detach().toList()
        break

    return sum(batch_losses) / len(batch_losses), sum(batch_predictions) / len(batch_predictions)

def _test_step(model, data_loader):
    batch_losses = []
    batch_predictions = []
    model.eval()
    with torch.no_grad():
        for _X, _y in data_loader:
            X, y = _X.to(DEVICE), _y.to(DEVICE)

            y_pred = model(X)

            # TODO why different loss then traning
            batch_loss = F.cross_entropy(y_pred, y, reduction='sum')
            batch_prediction = y_pred.argmax(dim=1, keepdim=True)

    return batch_losses.mean(), batch_predictions.mean()


if __name__ == '__main__':
    main(get_args())
