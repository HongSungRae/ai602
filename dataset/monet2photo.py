from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Monet2Photo(Dataset):
    def __init__(self, split, size=256):
        assert split in ['train', 'test']
        self.split = split
        self.folder_name_monet = split + 'A'
        self.folder_name_photo = split + 'B'
        self.data_list_monet = os.listdir(f'E:/sungrae/data/monet2photo/{self.folder_name_monet}')
        self.data_list_photo = os.listdir(f'E:/sungrae/data/monet2photo/{self.folder_name_photo}')
        self.transform = A.Compose([A.Resize(size,size),
                                    ToTensorV2()])
    
    def __len__(self):
        return min(len(self.data_list_monet), len(self.data_list_photo))
    
    def __getitem__(self, idx):
        # Monet
        img_monet = plt.imread(f'E:/sungrae/data/monet2photo/{self.folder_name_monet}/{self.data_list_monet[idx]}')/255 # (256,256,3)
        img_monet = self.transform(image=img_monet)['image']

        # Photo
        img_photo = plt.imread(f'E:/sungrae/data/monet2photo/{self.folder_name_photo}/{self.data_list_photo[idx]}')/255 # (256,256,3)
        img_photo = self.transform(image=img_photo)['image']

        # sacle to [-1,1]
        img_monet = (img_monet-0.5)/0.5
        img_photo = (img_photo-0.5)/0.5
        return img_monet.type(torch.float32), img_photo.type(torch.float32)


if __name__ == '__main__':
    '''
    # train
    G : 이미지[-1,1] 받아서 이미지[-1,1] 생성
    D : 이미지[-1,1] 받아서 예측[0,1]
    # test
    G : 이미지[-1,1] 받아서 이미지[-1,1] 생성
    '''
    dataset = Monet2Photo('train')
    dataloader = DataLoader(dataset, 32)
    img_monet, img_photo = next(iter(dataloader))
    print(img_monet.shape, img_photo.shape)
    print(torch.min(img_monet), torch.max(img_monet))

    del dataset, dataloader, img_monet, img_photo