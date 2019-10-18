import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.utils.data
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform


def get_train_dataloader(name, args):
    if name == 'DIV2K':
        dataset = DIV2K(args)
    if name == 'GoPro':
        dataset = GoProDataset('./data_meta/train_blur_file.txt', './data_meta/train_sharp_file.txt', './data', True,
                               multi_scale=True, rotation=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
    else:
        raise Exception('Dataset is not supported')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                             pin_memory=True)
    return dataloader


def get_test_dataloader(name, args):
    if name == 'Set5':
        dataset = Set5(args)
    elif name == 'Set14':
        dataset = Set14(args)
    elif name == 'B100':
        dataset = B100(args)
    elif name == 'Urban100':
        dataset = Urban100(args)
    elif name == 'GoPro':
        dataset = GoProDataset('./data_meta/test_blur_file.txt', './data_meta/test_sharp_file.txt', './data', crop=False,
                               multi_scale=True, rotation=False, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        raise Exception('Dataset is not supported')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads),
                                             pin_memory=False)
    return dataloader


class GoProDataset(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, root_dir, crop=False, crop_size=256, multi_scale=False,
                 rotation=False, color_augment=False, transform=None, ):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)
        self.rotate45 = transforms.RandomRotation(45)

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_image = Image.open(
            os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2], image_name[3])).convert('RGB')
        sharp_image = Image.open(
            os.path.join(self.root_dir, image_name[0], image_name[1], 'sharp', image_name[3])).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree)
            sharp_image = transforms.functional.rotate(sharp_image, degree)

        if self.color_augment:
            # contrast_factor = 1 + (0.2 - 0.4*np.random.rand())
            # blur_image = transforms.functional.adjust_contrast(blur_image, contrast_factor)
            # sharp_image = transforms.functional.adjust_contrast(sharp_image, contrast_factor)
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2]

            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]

            blur_image = blur_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            sharp_image = sharp_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]

        if self.multi_scale:
            H = sharp_image.size()[1]
            W = sharp_image.size()[2]
            blur_image_s1 = transforms.ToPILImage()(blur_image)
            sharp_image_s1 = transforms.ToPILImage()(sharp_image)
            blur_image_s2 = transforms.ToTensor()(transforms.Resize([H // 2, W // 2])(blur_image_s1)).mul(1.0)
            sharp_image_s2 = transforms.ToTensor()(transforms.Resize([H // 2, W // 2])(sharp_image_s1)).mul(1.0)
            blur_image_s3 = transforms.ToTensor()(transforms.Resize([H // 4, W // 4])(blur_image_s1)).mul(1.0)
            sharp_image_s3 = transforms.ToTensor()(transforms.Resize([H // 4, W // 4])(sharp_image_s1)).mul(1.0)
            blur_image_s1 = transforms.ToTensor()(blur_image_s1).mul(1.0)
            sharp_image_s1 = transforms.ToTensor()(sharp_image_s1).mul(1.0)

            # normalization [-1,1]
            # blur_image_s1 = (blur_image_s1 / 255.0 - 0.5) * 2
            # sharp_image_s1 = (sharp_image_s1 / 255.0 - 0.5) * 2
            # blur_image_s2 = (blur_image_s2 / 255.0 - 0.5) * 2
            # sharp_image_s2 = (sharp_image_s2 / 255.0 - 0.5) * 2
            # blur_image_s3 = (blur_image_s3 / 255.0 - 0.5) * 2
            # sharp_image_s3 = (sharp_image_s3 / 255.0 - 0.5) * 2

            return {'blur_image_s1': blur_image_s1, 'blur_image_s2': blur_image_s2, 'blur_image_s3': blur_image_s3,
                    'sharp_image_s1': sharp_image_s1, 'sharp_image_s2': sharp_image_s2,
                    'sharp_image_s3': sharp_image_s3}
        else:
            return {'blur_image': blur_image, 'sharp_image': sharp_image}


class Set14(data.Dataset):
    def __init__(self, args):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class B100(data.Dataset):
    def __init__(self, args):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class Urban100(data.Dataset):
    def __init__(self, args):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def np_to_tensor(imgIn, imgTar, channel):
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        imgTar = np.sum(imgTar * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2
    imgTar = (imgTar / 255.0 - 0.5) * 2
    return imgIn, imgTar


def augment(imgIn, imgTar):
    if random.random() < 0.3:  # horizontal flip
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if random.random() < 0.3:  # vertical flip
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]

    rot = random.randint(0, 3)  # rotate
    imgIn = np.rot90(imgIn, rot, (0, 1))
    imgTar = np.rot90(imgTar, rot, (0, 1))

    return imgIn, imgTar


def get_path(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize  # HR image patch size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    return imgIn, imgTar


class DIV2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        apath = args.dataDir
        dirHR = 'DIV2K_train_HR_crop_100'
        dirLR = 'DIV2k_train_LR_bicubic_crop_100/X2'
        self.dirIn = os.path.join(apath, dirLR)
        self.dirTar = os.path.join(apath, dirHR)
        self.fileList = os.listdir(self.dirTar)
        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):
        scale = self.scale
        args = self.args
        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)
        if self.args.need_patch:
            imgIn, imgTar = get_path(imgIn, imgTar, self.args, scale)
        imgIn, imgTar = augment(imgIn, imgTar)
        return np_to_tensor(imgIn, imgTar, self.channel)

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        name = name[0:-4] + 'x2' + '.png'
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar


class Set5(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        dirHR = args.HR_valDataroot
        dirLR = args.LR_valDataroot
        self.dirIn = os.path.join(dirLR)
        self.dirTar = os.path.join(dirHR)
        self.fileList = os.listdir(self.dirTar)
        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):
        scale = self.scale
        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)

        return np_to_tensor(imgIn, imgTar, self.channel)

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        name = name[0:-4] + 'x2' + '.png'
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar
