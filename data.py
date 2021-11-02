from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100



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


if __name__ == '__main__':
    tr, ts = get_cifar()
