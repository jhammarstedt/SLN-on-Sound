import torch
import numpy as np
import torch.nn.functional as F
import logging as log

from training_helpers import SLN, BatchRecord

def _train_step(args, model, data_loader, optimizer, momenturm_optimizer, device):
    log.debug('Train step ...')
    br = BatchRecord(device, data_loader)
    model.train()
    for _X, _y in data_loader:
        optimizer.zero_grad()

        X, y = _X.to(device), _y.to(device)

        # add stochastic label noise
        y = SLN(y, device, args.sigma)

        y_pred = model(X)
        loss_per_input = -torch.sum(F.log_softmax(y_pred, dim=1) * y, dim=1)
        batch_loss = torch.mean(loss_per_input)

        # backpropagation
        batch_loss.backward()
        optimizer.step()
        momenturm_optimizer.step()

        br.update(X, y, y_pred, torch.sum(loss_per_input))

    return br.get_record()


def _test_step(model, data_loader, device):
    log.debug('Test step ...')
    batch_losses = []
    batch_predictions = []
    br = BatchRecord(device, data_loader)
    model.eval()
    with torch.no_grad():
        for _X, _y in data_loader:
            X, y = _X.to(device), _y.to(device)
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            br.update(X, y, y_pred, loss*y.size(0))

    return br.get_record()


# Training loops for sound data
def _train_step_sound(args, model, data_loader, optimizer, momenturm_optimizer, device):
    log.debug('Train step ...')
    br = BatchRecord(device, data_loader)
    model.train()
    for _X, _, _y in tqdm(data_loader):
        optimizer.zero_grad()
        if len(_y.size()) == 1:
            one_hot = torch.tensor(np.eye(args.num_class)[_y], device=device)
        else:
            one_hot = _y

        X, y = _X.to(DEVICE), one_hot.to(device)

        # add stochastic label noise
        y = SLN(y, device, args.sigma)

        with torch.cuda.amp.autocast():
            y_pred = model(X)
            loss_per_input = -torch.sum(F.log_softmax(y_pred, dim=1) * y, dim=1)
            batch_loss = torch.mean(loss_per_input)

        # backpropagation
        batch_loss.backward()
        optimizer.step()
        momenturm_optimizer.step()

        br.update(X, y, y_pred, torch.sum(loss_per_input))

    return br.get_record()


def _test_step_sound(model, data_loader, device):
    log.debug('Test step ...')
    br = BatchRecord(device, data_loader)
    model.eval()
    with torch.no_grad():
        for _X, _y in tqdm(data_loader):
            X, y = _X.to(device), _y.to(device)
            y_pred = model(X).mean(0).unsqueeze(0)
            loss = F.cross_entropy(y_pred, y)
            br.update(X, y, y_pred, loss*y.size(0))

    return br.get_record()
