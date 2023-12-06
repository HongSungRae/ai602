from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .parameters import data_root



class AppleOrange(Dataset):
    def __init__(self, split, size=256):
        assert split in ['train', 'test']
        self.split = split
        self.folder_name_apple = split + 'A'
        self.folder_name_orange = split + 'B'
        self.data_list_apple = os.listdir(fr'{data_root}/apple2orange/{self.folder_name_apple}')
        self.data_list_orange = os.listdir(fr'{data_root}/apple2orange/{self.folder_name_orange}')
        self.transform = A.Compose([A.Resize(size,size),
                                    ToTensorV2()])
    
    def __len__(self):
        return min(len(self.data_list_apple), len(self.data_list_orange))
    
    def __getitem__(self, idx):
        # Apple
        img_apple = plt.imread(f'{data_root}/apple2orange/{self.folder_name_apple}/{self.data_list_apple[idx]}')/255 # (256,256,3), np.array
        img_apple = self.transform(image=img_apple)['image'] # (3,256,256), torch.tensor

        # Orange
        img_orange = plt.imread(f'{data_root}/apple2orange/{self.folder_name_orange}/{self.data_list_orange[idx]}')/255 # (256,256,3), np.array
        img_orange = self.transform(image=img_orange)['image'] # (3,256,256), torch.tensor

        # normalize to [-1,1]
        img_apple = (img_apple-0.5)/0.5
        img_orange = (img_orange-0.5)/0.5
        return img_apple.type(torch.float32), img_orange.type(torch.float32)


if __name__ == '__main__':
    '''
    # train
    G : 이미지[-1,1] 받아서 이미지[-1,1] 생성
    D : 이미지[-1,1] 받아서 예측[0,1]
    # test
    G : 이미지[-1,1] 받아서 이미지[-1,1] 생성
    '''
    dataset = AppleOrange('test')
    dataloader = DataLoader(dataset, 32)
    img_apple, img_orange = next(iter(dataloader))
    print(img_apple.shape, img_orange.shape)
    print(f'({torch.min(img_apple)}, {torch.max(img_orange)})')

    del dataset, dataloader, img_apple, img_orange