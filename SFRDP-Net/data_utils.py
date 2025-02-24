import os.path as osp
import pickle
import lmdb
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import albumentations as A
import os
import random
from torchvision.transforms.functional import rotate
from albumentations import functional as F

def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G) dtype:uint8
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img


def uint2tensor3(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor: torch.Tensor = torch.from_numpy(np.copy(img)).permute(
        2, 0, 1).float().div(255.)
    return tensor

def center_crop(image, target_size):
    h, w = image.shape[:2]
    th, tw = target_size
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return image[max(y1, 0):min(y1 + th, h), max(x1, 0):min(x1 + tw, w)]

class mydataset(Dataset):
    def __init__(self, haze_root, clear_root):
        super(mydataset, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        '''
        self.hazy_img_path = [os.path.join(haze_root, img) for img in os.listdir(haze_root)]
        #DHID
        #self.clear_img_path = [os.path.join(clear_root, img.split('_')[0]+'.jpg') for img in os.listdir(haze_root)]
        #StateHaze1K
        self.clear_img_path = [os.path.join(clear_root, img) for img in os.listdir(haze_root)]


    def __getitem__(self, idx):
        hazy_img = imread_uint(self.hazy_img_path[idx], 3)
        clear_img = imread_uint(self.clear_img_path[idx], 3)
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        return [hazy_img, clear_img]

    def __len__(self):
        return len(self.hazy_img_path)


def data2lmdb(dpath, haze_root, clear_root, lmdb_name, write_frequency=1000, num_workers=8):
    '''
    :param dpath:           存放lmdb的路径
    :param haze_root:       存放雾霾图像数据的路径
    :param clear_root:      存放清晰图像数据的路径
    :param lmdb_name:       保存文件的名字
    :param write_frequency: 越大越好
    :param num_workers:     越大越好
    :return:
    '''
    dataset = mydataset(haze_root, clear_root)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)
    lmdb_path = osp.join(dpath, "%s.lmdb" % lmdb_name)
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        haze, clear = data[0]
        temp = LMDB_Image(haze, clear)
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(temp))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    print("Flushing database ...")
    db.sync()
    db.close()


class LMDB_Image:
    def __init__(self, haze, clear):
        self.channels = haze.shape[2]
        self.size = haze.shape[:2]
        self.haze = haze.tobytes()
        self.clear = clear.tobytes()

    def get_image(self):
        """ Returns the image as a numpy array. """
        haze = np.frombuffer(self.haze, dtype=np.uint8)
        clear = np.frombuffer(self.clear, dtype=np.uint8)
        return haze.reshape(*self.size, self.channels), clear.reshape(*self.size, self.channels)


    
class down_DatasetLMDB(Dataset):
    def __init__(self, db_path, crop_size=256):
        '''
        :param db_path:     lmdb文件路径
        :param crop_size:   裁剪大小
        '''
        self.db_path = db_path
        self.crop_size = crop_size
        self.downscale_factors = [0.5, 0.7, 1.0]
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def aug_data(self, data, target):
        downscale_factor = random.choice(self.downscale_factors)
        data = self.downsample(data, downscale_factor)
        target = self.downsample(target, downscale_factor)

        # 如果图像已经是目标大小，跳过裁剪
        if data.shape[0] == self.crop_size and data.shape[1] == self.crop_size:
            # 不需要随机裁剪
            trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Flip(p=0.5),
            ], additional_targets={'target': 'image'})
            augmented = trans(image=data, target=target)
            data = augmented['image']
            target = augmented['target']
            data = uint2tensor3(data)
            target = uint2tensor3(target)
            return data, target

        trans = A.Compose([
            A.RandomCrop(self.crop_size, self.crop_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Flip(p=0.5),
        ], additional_targets={'target': 'image'})
        augmented = trans(image=data, target=target)
        data = augmented['image']
        target = augmented['target']
        data = uint2tensor3(data)
        target = uint2tensor3(target)
        return data, target

    def downsample(self, img, scale):
        '''通过指定比例下采样图像'''
        height, width = img.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE = pickle.loads(byteflow)
        hazy_img, clear_img = IMAGE.get_image()
        hazy_img, clear_img = self.aug_data(hazy_img, clear_img)
        return hazy_img, clear_img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'    

class DatasetLMDB(Dataset):
    def __init__(self, db_path, crop_size=256):
        '''
        :param db_path:     lmdb文件路径
        :param crop_size:   裁剪大小
        '''
        self.db_path = db_path
        self.crop_size = crop_size
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def aug_data(self, data, target):
        trans = A.Compose([
            A.RandomCrop(self.crop_size, self.crop_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0)
            # A.VerticalFlip(p=0.5),
        ], additional_targets={'target': 'image'})
        augmented = trans(image=data, target=target)
        data = augmented['image']
        target = augmented['target']
        data = uint2tensor3(data)
        target = uint2tensor3(target)
        return data, target

    def __getitem__(self, index):
        with self.env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE = pickle.loads(byteflow)
        hazy_img, clear_img = IMAGE.get_image()
        hazy_img, clear_img = self.aug_data(hazy_img, clear_img)

        return hazy_img, clear_img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class TestLMDB(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE = pickle.loads(byteflow)
        hazy_img, clear_img = IMAGE.get_image()
        hazy_img = uint2tensor3(hazy_img)
        clear_img = uint2tensor3(clear_img)
        return hazy_img, clear_img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def get_dataloader(opt):
    if not os.path.exists(opt.dpath + "/" + opt.train_lmdb_name + ".lmdb"):
        data2lmdb(opt.dpath,opt.train_haze_root,opt.train_clear_root,opt.train_lmdb_name,num_workers=opt.num_workers)
    if not os.path.exists(opt.dpath + "/" + opt.test_lmdb_name + ".lmdb"):
        data2lmdb(opt.dpath,opt.test_haze_root,opt.test_clear_root,opt.test_lmdb_name,num_workers=opt.num_workers)
    trainDataset = DatasetLMDB(opt.train_db_path, opt.crop_size)
    testDataset = TestLMDB(opt.test_db_path)
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, test_loader

# 以上为有监督图像修复数据集处理。
# 以下为无监督图像修复数据集处理。
class mydataset2(Dataset):
    def __init__(self, haze_root, clear_root,crop_size=None):
        super(mydataset2, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        '''
        self.crop_size = crop_size
        self.hazy_img_path = [os.path.join(haze_root, img) for img in os.listdir(haze_root)]
        self.clear_img_path = [os.path.join(clear_root, img) for img in os.listdir(haze_root)]
        self.len = len(self.clear_img_path)
        self.all_indices = list(range(len(self.clear_img_path)))


    def __getitem__(self, idx):
        hazy_img = imread_uint(self.hazy_img_path[idx], 3)
        c_idx = np.random.choice(self.all_indices)
        while c_idx == idx:
            c_idx = np.random.choice(self.all_indices)
        clear_img = imread_uint(self.clear_img_path[c_idx], 3)
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        if self.crop_size is not None:
            hazy_img = self.aug_data(hazy_img)
            clear_img = self.aug_data(clear_img)
        else:
            hazy_img = uint2tensor3(hazy_img)
            clear_img = uint2tensor3(clear_img)
        return hazy_img, clear_img

    def __len__(self):
        return len(self.hazy_img_path)


    def aug_data(self, data):
        trans = A.Compose([
            A.RandomCrop(self.crop_size, self.crop_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Flip(p=0.5),
        ])
        augmented = trans(image=data)
        data = augmented['image']
        data = uint2tensor3(data)
        return data

def get_dataloader2(opt):
    trainDataset = mydataset2(opt.train_haze_root,opt.train_clear_root, opt.crop_size)
    testDataset = mydataset2(opt.test_haze_root,opt.test_clear_root)
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, test_loader
