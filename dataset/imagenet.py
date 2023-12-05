from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch
from dataset import data_root

# mini : 
# tiny : https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
class ImageNet(Dataset):
    def __init__(self, dtype, split, size=64):
        assert dtype in ['mini','tiny'], 'ImageNet data type Error.'
        assert split in ['train', 'test', 'validation'], 'Split Error.'
        if dtype == 'mini':
            self.split = split if split=='train' else 'val'
        else:
            self.split = split
        self.dtype = dtype
        if dtype == 'mini':
            path = fr'{data_root}/imagenet-mini'
        else:
            path = fr'{data_root}/imagenet-tiny'
        self.df = self._make_split_info(path, self.split)
        if split == 'train':
            self.transform = A.Compose([A.Resize(size,size),
                                        A.HorizontalFlip(p=0.33),
                                        A.VerticalFlip(p=0.33),
                                        A.RandomCrop(size,size,p=0.33),
                                        A.Normalize(max_pixel_value=255),
                                        ToTensorV2()])
        elif split in ['test', 'validation', 'val']:
            self.transform = A.Compose([A.Resize(size,size),
                                        A.Normalize(max_pixel_value=255),
                                        ToTensorV2()])

    def _make_split_info(self, path, split):
        df_path = f'{path}/{split}.csv'
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            img_path_list = []
            img_class_list = []
            folder_list = sorted(os.listdir(f'{path}/{split}'))
            for idx, folder_name in enumerate(folder_list):
                folder_path = os.path.join(path, split, folder_name)
                imgs_in_folder = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
                img_path_list.extend(imgs_in_folder)
                classes = [idx] * len(imgs_in_folder)
                img_class_list.extend(classes)
            df = pd.DataFrame(data={'path':img_path_list, 'class':img_class_list})
            df = df.sample(frac=1)  # row 전체 shuffle
            df = df.sample(frac=1).reset_index(drop=True)  # shuffling하고 index reset
            df.to_csv(df_path, index=False)
            del img_class_list, img_path_list, folder_list
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, cls = self.df.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = self.transform(image=img)['image']
        
        return img.type(torch.float32), torch.tensor(int(cls)).type(torch.LongTensor)


if __name__ == '__main__':
    dataset = ImageNet('mini', 'test', 64)
    dataloader = DataLoader(dataset, 32, True)
    img, target = next(iter(dataloader))

    print(img.shape, target.shape) # (b,3,size,size), (b,)
    print(torch.min(img), torch.max(img))