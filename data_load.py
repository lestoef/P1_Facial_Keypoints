import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class NormalizeRGB(object):
    """Convert a color image to GBR and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
        
        # mean and standard deviation used for color channel normalization
        norm_means = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]      
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # normalize color channels
        image_copy = ((image_copy/255.0 - norm_means) / norm_std).clip(0, 1)
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
    
class ToTensorRGB(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        # Ensure the image data type is float32
        image = image.astype(np.float32)
        
        # if image has no color channel, add one
        if len(image.shape) == 2:
            # Add the third color dimension
            image = image.reshape(image.shape[0], image.shape[1], 3)
            
        # Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Mean and standard deviation used for color channel normalization
        means = [image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()]
        stds = [image[:,:,0].std(), image[:,:,1].std(), image[:,:,2].std()]
        
        # Normalize the image
        image = ((image - means) / stds) / 255.
        
        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        # Scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts = (key_pts - 100) / 50.0
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
    
class FaceCrop(object):
    ''' inspired by https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4'''
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        # get outer dimensions of face based on keypoints
        left = key_pts.transpose()[0].min()
        right = key_pts.transpose()[0].max()
        bottom = key_pts.transpose()[1].min()
        top = key_pts.transpose()[1].max()
        kp_width = right-left
        kp_height = top-bottom
        # assume the face to be 10% bigger than the keypoint outline in each direction
        width_padding = kp_width*0.1
        height_padding = kp_height*0.1

        # create crop coordinates
        h, w = image.shape[:2]
        new_left = max(0, int(left-width_padding))
        new_right = min(int(right+width_padding), w)
        new_bottom = max(0, int(bottom-height_padding))
        new_top = min(int(top+height_padding), h)

        # crop image
        image = image[new_bottom: new_top,
                      new_left: new_right]
        # adapt keypoint coordinates
        key_pts = key_pts - [new_left, new_bottom]

        return {'image': image, 'keypoints': key_pts}
