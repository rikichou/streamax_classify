# ------------------------------------------------------------------------------
# driver fatigue clssificaiton
# Copyright (c) Streamax.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Hong Hu (huhong@streamax.com)
# dataloader
# ------------------------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import logging
# https://askubuntu.com/questions/742782/how-to-install-cpickle-on-python-3-4
import _pickle as pickle
import cv2
from PIL import Image
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import common
import matplotlib as plt
import random


class DatasetFacialExpression(Dataset):
    def __init__(self, data_prefix, ann_file, transform=None, target_transform=None,
                 cache_root='data/cache', insize=300, if_bgr=True, train=True,
                 require_path=False, mode='color'):
        self.data_prefix = data_prefix
        # [Cache Point]
        self.cache_root = cache_root
        self.samples = make_dataset_Expression(self.data_prefix, ann_file)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_prefix))

        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.if_bgr = if_bgr
        self.require_path = require_path

        if if_bgr:
            self.preproc = preproc
            self.mean = (104, 117, 123)
            self.insize = insize
            self.loader = opencv_loader_withrect
            self.tarin = train
            self.mode = mode
        # preproc(image, mean, insize)
        # self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.

        info = {'img_prefix': data_prefix}
        info['img_info'] = {'filename': filename}
        info['gt_label'] = gt_label
        info['face_rect'] = np.array([sx,sy,ex,ey], dtype=np.int64)
        """
        info = self.samples[index]
        path = os.path.join(info['img_prefix'], info['img_info']['filename'])
        target = info['gt_label']
        face_rect = info['face_rect']
        # TODO: Convert RGB to Gray Color
        sample = self.loader(path, face_rect, self.mode)
        # print(sample.shape)

        if not self.if_bgr:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            sample = self.preproc(sample, self.mean, self.insize, self.tarin, path, self.mode)

        if self.require_path:
            return sample, target, path

        return sample, target

class DatasetPlayPhone(Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, extensions=None, 
            cache_root='data/cache', insize = 300, if_bgr = True, train = True,
            require_path = False, use_cache=False, mode='color'):
        self.root = root
        self.image_set = image_set
        self.classes, self.class_to_idx = self._find_classes(self.root)
        # [Cache Point]
        self.cache_root = cache_root
        if use_cache:
            db_file = os.path.join(self.cache_root, 'PlayPhone_cached_{}_db.pkl'.format(self.image_set))
            if os.path.exists(db_file):
                logger.info('=> load data from {}'.format(db_file))
                with open(db_file, 'rb') as fd:
                    self.samples = pickle.load(fd)
            else:
                self.samples = make_dataset_PlayPhone(self.root, self.class_to_idx, extensions=extensions)
                os.makedirs(self.cache_root, exist_ok=True)
                logger.info('=> save data to {}'.format(db_file))
                with open(db_file, 'wb') as fd:
                    pickle.dump(self.samples, fd)
        else:
            self.samples = make_dataset_PlayPhone(self.root, self.class_to_idx, extensions=extensions)

        # self.samples = _make_dataset(self.root, self.class_to_idx, extensions=extensions)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root +
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.if_bgr = if_bgr
        self.require_path = require_path
        
        if if_bgr:
            self.preproc = preproc
            self.mean = (104, 117, 123)
            self.insize = insize
            self.loader = opencv_loader_new
            self.tarin = train
            self.mode = mode
        #preproc(image, mean, insize)
        #self.targets = [s[1] for s in self.samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        '''
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # '''
        # class_to_index = {'beijing':0, 'chengke':1, 'siji':2, 'beijing_half_1':0, 'chengke_half_1':1, 
        #     'siji_half_1':2, 'beijing_half_2':0, 'chengke_half_2':1, 'siji_half_2':2,
        #     'beijing_res':0, 'chengke_res':1, 'siji_res':2}
        # class_to_index = {'chengke':0, 'siji':1, 'chengke_half_1':0, 'siji_half_1':1, 'chengke_half_2':0, 'siji_half_2':1}
        # class_to_index = {'beijing':0, 'chengke':1, 'siji':2, 'chengke_augumented':1}

        class_to_index = {'beijing':0, 'chengke':1, 'siji':2, 
                'beijing_wrong':0, 'chengke_wrong':1, 'siji_wrong':2,
                'beijing_add':0, 'chengke_add':1, 'siji_add':2}

        # class_to_index = {'background':0, 'phone':1, 'smoke':2}
        # class_to_index = {'background':0, 'phone':1}
        classes = [d.name for d in os.scandir(dir) if d.name in class_to_index.keys()]
        class_to_idx = {key: class_to_index[key] for key in classes} 
        logger.info('=> dataset category {}'.format(class_to_idx))
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # TODO: Convert RGB to Gray Color
        sample = self.loader(path, self.mode)
        # print(sample.shape)

        if not self.if_bgr:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            sample = self.preproc(sample, self.mean, self.insize, self.tarin, path, self.mode)

        if self.require_path:
            return sample, target, path

        return sample, target

def getlist(dir, extension):
    """get a list of all files based on the specified file suffix name, like .jpg/.pts etc.
    :param dir:
        root diectory path.
    :param extension:
        specified file suffix name, like .jpg/.pts etc.
    :return:
        files list.
    """
    files_list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            if has_file_allowed_extension(name, extensions=extension):
                files_list.append(name)
    return files_list


def get_features(data_dir, feature_len=30):
    sequence = []
    feat_files = getlist(os.path.join(data_dir, data_dir), '.align')
    if len(feat_files) != feature_len:
        False, sequence
    for idx in range(feature_len):
        feature = open(os.path.join(data_dir, sequence, '{}.align'.format(idx))).read()
        feature = np.array([float(num) for num in feature.split(',')])
        sequence.append(feature.reshape(-1))
    print(sequence.shape)
    return True, sequence


def get_feature(fname):
    feature = open(os.path.join(fname)).read()
    feature = np.array([float(num) for num in feature.split(',')])
    return feature


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None, is_reg_bbox=False):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        target_class_num = 0
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    target_class_num += 1

        logger.info('=> load {} with target {} and {} samples'.format(target_class, class_index, target_class_num))
    return instances


def _make_dataset(directory, class_to_idx, extensions=None, is_valid_folder=None, check_len=30):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_folder is None
    both_something = extensions is not None and is_valid_folder is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_folder(x):
            return has_folder_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        target_class_num = 0
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        # for root, dirs, _ in sorted(os.walk(target_dir, followlinks=True)):
        dirs = sorted(os.listdir(target_dir))
        for fdir in tqdm(sorted(dirs)):
            path = os.path.join(target_dir, fdir)
            if len(getlist(path, extensions)) < check_len:
                continue
            features = []
            for idx in range(check_len):
                fname = '{}/{}.{}'.format(path, idx, extensions)
                features.append(get_feature(fname))
            features = np.array(features)
            if features.ndim == 2:
                features = features[None, :, :]
            item = features, class_index
            instances.append(item)
            target_class_num += 1

        logger.info('=> load {} with target {} and {} samples'.format(target_class, class_index, target_class_num))
        # print('=> load {} with target {} and {} samples'.format(target_class, class_index, target_class_num))
    return instances



def make_dataset_Expression(data_prefix, ann_file):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'Angry',
        'Happy',
        'Neutral',
        'Sad'
    ]

    with open(ann_file) as f:
        samples = [x.strip().split(';') for x in f.readlines()]

    statistics = {}
    data_infos = []
    for filename, gt_label, sx, sy, ex, ey in samples:
        gt_label = int(gt_label)
        sx = int(sx)
        sy = int(sy)
        ex = int(ex)
        ey = int(ey)
        info = {'img_prefix': data_prefix}
        info['img_info'] = {'filename': filename}
        info['gt_label'] = gt_label
        info['face_rect'] = np.array([sx,sy,ex,ey], dtype=np.int64)
        data_infos.append(info)

        # statistics
        if gt_label not in statistics:
            statistics[gt_label] = 0
        statistics[gt_label] += 1

    print("!!!! data statistics ", statistics)
    return data_infos

def make_dataset_PlayPhone(directory, class_to_idx, extensions=None, is_valid_file=None, is_reg_bbox=False):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        target_class_num = 0
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    # TODO: add load bbox in item
                    item = path, class_index
                    instances.append(item)
                    target_class_num += 1

        logger.info('=> load {} with target {} and {} samples'.format(target_class, class_index, target_class_num))
    return instances

def has_folder_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename is folder, and extensions is folder type
    """
    return os.path.isdir(filename) and extensions == 'folder'


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def default_loader(path, default_loader='opencv', rgb_input=False):
    '''
    if default_loader == 'opencv':
        return opencv_loader(path, rgb_input=rgb_input)

    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    '''
    return pil_loader(path)
    
def opencv_loader(path, channel=1, rgb_input=False):
    # cv2.imdecode(np.fromfile( img_paths, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATIO
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), channel)
    if rgb_input:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return Image.fromarray(data)

def opencv_loader_new(path, mode='color'):
    if mode == 'color':
        data = cv2.imread(path, cv2.IMREAD_COLOR)
    elif mode == 'gray':
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Image read mode is not supported')
        return

    return data

def get_input_face(image, rect):
    sx,sy,ex,ey = rect
    if len(image.shape)<3:
        h,w = image.shape
    else:
        h,w,c = image.shape
    faceh = ey-sy
    facew = ex-sx

    longsize = max(faceh, facew)
    expendw = longsize-facew
    expendh  = longsize-faceh

    sx = sx-(expendw/2)
    ex = ex+(expendw/2)
    sy = sy-(expendh/2)
    ey = ey+(expendh/2)

    sx = int(max(0, sx))
    sy = int(max(0, sy))
    ex = int(min(w-1, ex))
    ey = int(min(h-1, ey))

    if len(image.shape)<3:
        return image[sy:ey, sx:ex]
    else:
        return image[sy:ey, sx:ex, :]

def opencv_loader_withrect(path, rect, mode='color'):
    if mode == 'color':
        data = cv2.imread(path, cv2.IMREAD_COLOR)
    elif mode == 'gray':
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Image read mode is not supported')
        return

    assert data != None, "failed to poen {}".format(path)
    data = get_input_face(data, rect)
    return data

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # NOTE: PIL.convert https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def yuv_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # NOTE: PIL.convert https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('YCbCr')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def preproc(image, mean, insize, train, path=None, mode='color'):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    if train:
        try:
            image = _mirror(image)
            image = _gaussianBlur(image)
            # interp_method = random.choice(interp_methods)
            image = cv2.resize(image, (insize, insize),interpolation=cv2.INTER_LINEAR)
        except:
            print(path)
    else:
        try:
            image = cv2.resize(image, (insize, insize),interpolation=cv2.INTER_LINEAR)
        except:
            print(path)
    image = image.astype(np.float32)
    if mode == 'color':
        image -= mean
        return torch.from_numpy(image.transpose(2, 0, 1))
    elif mode == 'gray':
        image /= 255
        return torch.from_numpy(image).unsqueeze(0)
    else:
        print('Image read mode is not supported')
        return

def _mirror(image):
    if random.randrange(2):
        image = image[:, ::-1]

    return image

def _gaussianBlur(image):
    if random.randrange(2):
        kernel_size = (5, 5)
        sigma = 1.5
        image = cv2.GaussianBlur(image, kernel_size, sigma)
    return image

def get_paths(root, extensions='jpg'):
    paths = []
    for file_name in os.listdir(root):
        if not file_name.lower().endswith(extensions):
            continue
        file_path = os.path.join(root, file_name)
        paths.append(file_path)
    return paths

if __name__ == '__main__':
    get_features(data_dir='samples')
    train_dataset = DatasetRotation(
        root='samples',
        extensions='align',
        transform=None,
        target_transform=None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=True,
        pin_memory=True
    )

    for i, (input, target) in enumerate(train_loader):
        # TODO:input BSx1x30x512, target BS
        target = target.cuda(non_blocking=True)
        print(input.shape, target)

