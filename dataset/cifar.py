from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataset import data_root

def get_cifar_dataset(cifar=None, split=None, size=256):
    assert cifar in ['cifar10', 'cifar100']
    assert split in ['train', 'test']
    root = fr'{data_root}/'
    transform = transforms.Compose([transforms.Resize((size,size)),
                                    transforms.RandomCrop(32,padding=4),
                                    transforms.RandomVerticalFlip(p=0.33),
                                    transforms.RandomHorizontalFlip(p=0.33),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.Resize((size,size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                              std=(0.2023, 0.1994, 0.2010))])
    if cifar == 'cifar10':
        num_classes = 10
        if split == 'train':
            dataset = datasets.CIFAR10(root=root,
                                       train=True,
                                       download=True,
                                       transform=transform)
        elif split == 'test':
            dataset = datasets.CIFAR10(root=root,
                                       train=False,
                                       download=True,
                                       transform=test_transform)
    else: # cifar100
        num_classes = 100
        if split == 'train':
            dataset = datasets.CIFAR100(root=root,
                                        train=True,
                                        download=True,
                                        transform=transform)
        elif split == 'test':
            dataset = datasets.CIFAR100(root=root,
                                       train=False,
                                       download=True,
                                       transform=test_transform)
    return dataset, num_classes



if __name__ == '__main__':
    dataset, _ = get_cifar_dataset('cifar100', 'test')
    dataloader = DataLoader(dataset, 32, True)
    x, y = next(iter(dataloader))
    print(f'{x.shape}, {y.shape}') # [b, 3, size, size], [b]