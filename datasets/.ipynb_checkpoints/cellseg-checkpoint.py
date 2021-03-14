import os
import glob

import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms



class CellSeg(data.Dataset):
    def __init__(self, mode, data_root,fold,transforms=None):
        
        self.data_root = data_root
        self.mode = mode
        self.transforms = transforms
        self.fold = fold
        self.mapping = {
            0: 0,
            255: 1              
        }

        if mode == 'train':
            self.images_path = data_root + 'train_image_{}/output/'.format(self.fold)
            self.mask_path = data_root + 'train_image_{}/output/'.format(self.fold)
            self.images = glob.glob(self.images_path + "train_image_*")
            self.masks = glob.glob(self.images_path + "_groundtruth_(1)*")

        if mode == 'val':
            self.images_path = data_root + 'test_image_{}/*.png'.format(self.fold)
            self.mask_path = data_root + 'test_mask_{}/*.png'.format(self.fold)
            self.images = glob.glob(self.images_path)
            self.masks = glob.glob(self.mask_path)
            
        self.images.sort()
        self.masks.sort()
    
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask

    def __getitem__(self, index):
        
        image = Image.open(self.images[index])
        image = image.convert('L')

        mask = Image.open(self.masks[index])
        thresh = 127.5
        fn = lambda x : 255 if x > thresh else 0
        mask = mask.convert('L').point(fn, mode='1')
        mask = torch.from_numpy(np.asarray(mask, dtype=np.uint8)) # this is for my dataset(lv)
        mask = self.mask_to_class(mask)
        mask = mask.long()
        
        
        if self.transforms is not None:
            image = self.transforms(image)
#             mask = self.transforms(mask)

#         print(np.unique(mask.numpy()))
        return (image, mask)
    
    

    def __len__(self):
        return len(self.images)



class CellSegDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test', 'random']

        # mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        # transformations_train = transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor()])

        self.input_transform = transforms.Compose([transforms.ToTensor()])

        # self.target_transform = standard_transforms.Compose([
        #     standard_transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
        #     extended_transforms.MaskToTensor()
        # ])

        # self.restore_transform = standard_transforms.Compose([
        #     extended_transforms.DeNormalize(*mean_std),
        #     standard_transforms.Lambda(lambda x: x.div_(255)),
        #     standard_transforms.ToPILImage(),
        #     extended_transforms.FlipChannels()
        # ])

        # self.visualize = standard_transforms.Compose([
        #     standard_transforms.Resize(400),
        #     standard_transforms.CenterCrop(400),
        #     standard_transforms.ToTensor()
        # ])
        # if self.config.mode == 'random':
        #     train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
        #                              self.config.img_size)
        #     train_labels = torch.ones(self.config.batch_size, self.config.img_size, self.config.img_size).long()
        #     valid_data = train_data
        #     valid_labels = train_labels
        #     self.len_train_data = train_data.size()[0]
        #     self.len_valid_data = valid_data.size()[0]

        #     self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
        #     self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

        #     train = TensorDataset(train_data, train_labels)
        #     valid = TensorDataset(valid_data, valid_labels)

        #     self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
        #     self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        if self.config.mode == 'train':
            train_set = CellSeg('train', self.config.data_root,
                                fold = self.config.fold,
                                transforms=self.input_transform
                               )
            valid_set = CellSeg('val', self.config.data_root,
                                fold =self.config.fold,
                                transforms=self.input_transform
                               )

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size
            
        elif self.config.mode == 'test':
            valid_set = CellSeg('val', self.config.data_root,
                                fold =self.config.fold,
                                transforms=self.input_transform
                               )
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size
            

        # elif self.config.mode == 'test':
        #     test_set = VOC('test', self.config.data_root,
        #                    transform=self.input_transform, target_transform=self.target_transform)

        #     self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
        #                                   num_workers=self.config.data_loader_workers,
        #                                   pin_memory=self.config.pin_memory)
        #     self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass

    
    
# # train_set = CellSeg('train', "../../Dataset/",
# #                             transforms.Compose([transforms.ToTensor()]))

# valid_set = CellSeg('val', "../../Dataset/",
#                            transforms.Compose([transforms.ToTensor()]))
# valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

# # train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

# for image, mask in valid_loader:
#     print(mask.size())
    