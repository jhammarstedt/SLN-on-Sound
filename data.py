from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

import os
import numpy as np
import pandas as pd
import logging as log

log.basicConfig(level=log.DEBUG)

def get_cifar(dataset: str = 'cifar10',
              path: str = './data-cifar',
              labels_path: str = './labels',
              noise_mode: str = 'sym',
              noise_rate: float = 0.4):

    train_dataset = None
    test_dataset = None
    if dataset == 'cifar10':
        # normalization values taken from https://github.com/kuangliu/pytorch-cifar/issues/19
        # we use same transformation as described in Appendix B.1 of the original paper
        train_transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = CIFAR10(path, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(path, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        # normalization values taken from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
        # we use same transformation as described in Appendix B.1 of the original paper
        train_transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        train_dataset = CIFAR100(path, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(path, train=False, transform=test_transform, download=True)

    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported ATM')

    noisy_labels_path = os.path.join(labels_path, 'noisy_labels_' + dataset + '_' + noise_mode + '_' + str(noise_rate) + '.csv')

    if noise_mode == '':
        log.info('Getting clean dataset')

    elif noise_mode in ['asym', 'sym']:
        log.info(f'Getting dataset with {noise_mode}metric noise')
        if not os.path.exists(noisy_labels_path):
            generate_noisy_labels(dataset, labels_path, noise_mode, noise_rate, train_dataset=train_dataset)

        labels = pd.read_csv(noisy_labels_path, index_col=None)
        train_dataset.targets = labels['noisy'].to_list()

    else:
        raise NotImplementedError('Unknown noise mode')

    return train_dataset, test_dataset


def generate_noisy_labels(dataset='cifar10', path='./labels', noise_mode='sym', noise_rate=0.4, train_dataset=None):
    if train_dataset is None:
        train_dataset, _ = get_cifar(dataset)
    clean_labels = train_dataset.targets
    noisy_labels = []

    min_class = min(clean_labels)
    max_class = max(clean_labels)

    np.random.seed(42)
    if noise_mode == 'sym':
        for label in clean_labels:
            if np.random.uniform() < noise_rate:
                new_label = np.random.randint(low=min_class, high=max_class + 1)
                while new_label == label:
                    new_label = np.random.randint(low=min_class, high=max_class + 1)
                noisy_labels.append(new_label)
            else:
                noisy_labels.append(label)

    elif noise_mode == 'asym':
        if dataset == 'cifar10':
            # Only similar labels are flipped
            # labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            # number on each index corresponds to which class it should be switched
            # TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → HORSE, and CAT ↔ DOG
            label_mapping = [0, 1, 0, 5, 7, 3, 6, 7, 8, 1]
            for label in clean_labels:
                if np.random.uniform() < noise_rate:
                    noisy_labels.append(label_mapping[label])
                else:
                    noisy_labels.append(label)

        elif dataset == 'cifar100':
            # Labels are flipped in circularly (0->1, 1->2, ..., 9->0)
            N_labels = len(train_dataset.classes)
            for label in clean_labels:
                if np.random.uniform() < noise_rate:
                    noisy_labels.append((label + 1) % N_labels)
                else:
                    noisy_labels.append(label)

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    labels = pd.DataFrame({'clean': clean_labels, 'noisy': noisy_labels})
    log.info('Rate of noisy labels: {}'.format((labels.clean != labels.noisy).sum() / labels.shape[0]))
    labels.to_csv(os.path.join(path, 'noisy_labels_' + dataset + '_' + noise_mode + '_' + str(noise_rate) + '.csv'),
                  index=False)


if __name__ == '__main__':
    train, test = get_cifar(dataset='cifar10', noise_mode='asym')
    print()
