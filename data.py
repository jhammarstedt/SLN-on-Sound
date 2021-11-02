from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

import os
import numpy as np
import pandas as pd



def get_cifar(dataset: str='cifar10', path: str='data'):
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

    return train_dataset, test_dataset


def generate_noisy_labels(dataset='cifar10', path='./data', noise_mode='sym', noise_rate=0.4):
    train_dataset, _ = get_cifar(dataset, path)
    clean_labels = train_dataset.targets
    noisy_labels = []

    min_class = min(clean_labels)
    max_class = max(clean_labels)

    np.random.seed(42)
    if noise_mode == 'sym':
        for label in clean_labels:
            if np.random.uniform() < noise_rate:
                while (new_label := np.random.randint(low=min_class, high=max_class+1)) == label:
                    continue
                noisy_labels.append(new_label)
            else:
                noisy_labels.append(label)
    labels = pd.DataFrame({'clean': clean_labels, 'noisy': noisy_labels})
    print('Rate of noisy labels: {}'.format((labels.clean != labels.noisy).sum() / labels.shape[0]))
    labels.to_csv(os.path.join(path, 'noisy_labels_' + dataset + '_' + noise_mode + '_' + str(noise_rate) + '.csv'))


if __name__ == '__main__':
    generate_noisy_labels(dataset='cifar100')
