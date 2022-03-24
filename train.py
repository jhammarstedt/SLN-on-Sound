import time

import torch
import logging as log

from args import get_args
from helpers import WeightExponentialMovingAverage, TrainingLogger
from training_helpers import get_cifar_data, get_FSD_data, get_models, get_FSD_models, label_correction
from training_loops import _train_step, _test_step, _train_step_sound, _test_step_sound

from fsd50_src.src.utilities.config_parser import parse_config, get_data_info
from fsd50_src.src.data.transforms import get_transforms_fsd_chunks

"""
Sound data preprocessing and networks are credited to: https://github.com/SarthakYadav/fsd50k-pytorch
Minor edits and workarounds were added to the code to make it work as intended in this scenario

Wide ResNet structure as per the original code: https://github.com/chenpf1025/SLN
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    log.getLogger().setLevel(args.loglevel.upper())
    log.info(f'Using {DEVICE} for torch training.')

    if args.data_type == 'image':
        train_loader, train_set, test_loader, train_eval_loader = get_cifar_data(args)
        # original_train_Y = np.eye(args.num_class)[train_set.targets]
        original_train_Y = train_set.targets.copy()


        model, momentum_model = get_models(args, DEVICE)
        momentum_model.load_state_dict(model.state_dict())

        train_step, test_step = _train_step, _test_step
    elif args.data_type == 'sound':
        # Data configs for the sound data
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

        train_loader, train_set, test_loader, train_eval_loader = get_FSD_data(args)
        args.num_class = args.cfg['model']['num_classes']

        model, momentum_model = get_FSD_models(args)

        train_step, test_step = _train_step_sound, _test_step_sound

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

        # Check if it is time to start label correction
        if epoch >= args.correction:
            args.sigma = 0.
            label_correction(args, momentum_model, train_eval_loader, train_set, original_train_Y)

        # Perform train step and test on both momentum model and regular model
        train_loss, train_acc = train_step(args, model, train_loader, optimizer, optimizer_momentum, DEVICE)
        test_loss, test_acc = test_step(momentum_model, test_loader, DEVICE)
        test_loss_NoEMA, test_acc_NoEMA = test_step(model, test_loader, DEVICE)

        training_log.save_epoch(train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA)
        training_log.print_last_epoch(epoch=epoch, logger=log, time=time.time() - epoch_start)

    training_log.export_as_json(args.trainlog_path)


if __name__ == '__main__':
    main(get_args())
