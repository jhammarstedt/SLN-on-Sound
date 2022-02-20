import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--trainlog_path", type=str, default='training_log.json')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train on')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--path', type=str, default='data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--noise_mode', type=str, default='dependent', help='Noise mode')
    parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
    parser.add_argument('--sigma', type=float, default=0.5, help='STD of Gaussian noise')
    parser.add_argument("--stdev", type=float, default=0.5, help="How much added noise")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument("--resume_from", type=str, help="checkpoint path to continue training from")
    parser.add_argument("--cw", type=str, required=False, help="path to serialized torch tensor containing class weights")
    parser.add_argument("--pretrained", type=bool, default=False, help=" Set to true to use pretrained model")
    parser.add_argument("--momentum", type=float, default=0.9, help="...")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="...")
    parser.add_argument("--correction", type=int, default=20, help="...")
    # FSD50k specific
    parser.add_argument("--cfg_file", type=str, help='path to cfg file')
    parser.add_argument("--expdir", "-e", type=str, help="directory for logging and checkpointing")

    parser.add_argument('--loglevel', default='info', help='Provide logging level. Example --loglevel debug, default=warning')
    
    # For running abliation study
    parser.add_argument("--abliation",type=bool,default=False,help="Set to true to conduct abliation study")
    
    parser.add_argument('--parameter', type=str, default="sigma", 
    help="What parameter to change in the ablation study: only sigma so far")
    parser.add_argument('--runs', type=int, default=3, help='number of runs for ablation study')

    
    return parser.parse_args()
