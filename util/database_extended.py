from typing import Tuple, Callable, List
import albumentations as albu

import torch
import os
import numpy as np
import cv2
from PIL import Image, ImageOps,ImageFile

TARGET_SIZE = (256, 256)
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_mean=84.05340889841818 / 255.0
train_std=52.46762714475937 / 255.0
val_mean=75.09208509657118/ 255.0
val_std=34.349294501422754/ 255.0
test_mean=74.4865044755898/ 255.0
test_std=40.19805828709091/ 255.0

class ScleraSegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, root: str ='data_extended/', mode: str = "TRAIN", channel: int = 3, rotation_limit=0, elastic_transform_prob=0, set_names=[]) -> None:
        self.data_dir = root
        self.mode = mode
        # scan the data directory for all files
        self.files, self.classes, self.dirs, self.set_names = self.scan(self.data_dir, set_names)
        self.channel = channel

        self.augmentation = get_training_augmentation(rotation_limit=rotation_limit, elastic_transform_prob=elastic_transform_prob)


    def scan(self, dir, set_names) -> Tuple[List[str], List[str]]:
        files = []
        classes = []
        dirs = []
        set_name_list = []

        c = 0

        for set_name in set_names:

            ids = os.listdir(os.path.join(dir, set_name, 'input'))
            for id in ids:
                for f in os.listdir(os.path.join(dir, set_name, 'input', id)):
                    files.append(f)
                    classes.append(int(c))
                    dirs.append(int(id))
                    set_name_list.append(set_name)
                c += 1

        return files, classes, dirs, set_name_list


    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if the given index is out of bounds, throw an IndexError
        if idx >= len(self):
            raise IndexError

        # get the respective file name and class
        f = self.files[idx]
        c = self.dirs[idx]
        cls = self.classes[idx]
        set_name = self.set_names[idx]

        # load the input and label images
        input = cv2.imread(os.path.join(self.data_dir, set_name, 'input', str(c), f)) #Image.open(os.path.join(self.data_dir, 'input', str(c), f))

        if (self.channel==1):
            # input = ImageOps.grayscale(Image.fromarray(input))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2YUV)[:,:,0]
            input = np.expand_dims(input, axis=2)


        if self.mode.upper() in ['TRAIN', 'VAL']:
            label =  cv2.imread(os.path.join(self.data_dir, set_name, 'label', str(c), f))  #Image.open(os.path.join(self.data_dir, 'label', str(c), f))
            label = ImageOps.grayscale(Image.fromarray(label))
            label = np.expand_dims(label, axis=2)

        # Do augmentation for training data

        if self.mode.upper() == 'TRAIN':
            sample = self.augmentation(image=input, mask=label)
            input, label = sample['image'], sample['mask']
        '''
        if (self.channel==1):
         input= input / 255.0
         if self.mode.upper() == 'TRAIN':
            input= (input - train_mean) / train_std
         elif self.mode.upper() == 'VAL':
            input= (input - val_mean) / val_std
         else:
            input= (input - test_mean) / test_std
         # convert everything( to tensors and return it
         input = torch.tensor((self.to_tensor(input)))
        else:
        '''
        input = torch.tensor((self.to_tensor(input) / 255.0))


        if self.mode.upper() in ['TRAIN', 'VAL']:
            label = torch.tensor( self.to_tensor(np.round( label /255.0)) )
        c = torch.tensor(c)

        return (input, np.round(label), cls, f, set_name) if self.mode.upper() in ['TRAIN', 'VAL'] else (input, cls, f, set_name, self.dirs[idx])

    def to_tensor(self,x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def resize_img_with_border(self,im, desired_size=256):
         if(len(im.shape)==2):
            new_im=np.zeros((im.shape[0],im.shape[1],3))
            new_im[:,:,0]=im
            new_im[:,:,1]=im
            new_im[:,:,2]=im
            im=new_im

         old_size = im.shape[:2]  # old_size is in (height, width) format

         ratio = float(desired_size) / max(old_size)
         new_size = tuple([int(x * ratio) for x in old_size])

         # new_size should be in (width, height) format

         im = cv2.resize(im, (new_size[1], new_size[0]))

         delta_w = desired_size - new_size[1]
         delta_h = desired_size - new_size[0]
         top, bottom = delta_h // 2, delta_h - (delta_h // 2)
         left, right = delta_w // 2, delta_w - (delta_w // 2)

         #color = [0, 0, 0]
         new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
         return new_im


def get_training_augmentation(rotation_limit=0, elastic_transform_prob=False):

    basic_transform = [
        albu.RandomCrop(height= 236, width=236, p=0.5),
        albu.CenterCrop(height= 236, width=236, always_apply=False,p=0.5),
        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=cv2.BORDER_REPLICATE),
    ]

    spatial_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=rotation_limit, shift_limit=0.1, p=1, border_mode=0),

        albu.ElasticTransform(p=elastic_transform_prob),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
    ]


    pixel_transform = [
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1)
            ],
            p=1,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=1,
        )
    ]

    return albu.Compose(basic_transform + spatial_transform + pixel_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
