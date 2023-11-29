from torch.utils.data import Dataset, DataLoader



class ADE20K(Dataset):
    def __init__(self, split):
        assert split in ['train', 'test'], 'ADE20K data type Error.'


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    pass