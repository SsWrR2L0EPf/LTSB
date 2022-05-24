import random

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from queue import Queue
import pickle


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        # np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

        self.sample_index = []
        self.sample_ptr = [0] * self.cls_num
        for _ in range(self.cls_num):
            self.sample_index.append([])
        for i, label in enumerate(self.targets):
            self.sample_index[label].append(i)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # if self.transform is not None:
        sample1 = self.transform[0](img)
        sample2 = self.transform[1](img)

        return [sample1, sample2], target
        # return sample1, target


class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class IMBALANCECIFAR10_MNIST(Dataset):
    cls_num = 10

    def __init__(self, root='../datasets/imbalance_cifar10', imb_factor=0.01, train=True, transform=None, ):
        super(IMBALANCECIFAR10_MNIST, self).__init__()
        self.train = train
        self.transform_val_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform_val_cifar = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform = transform
        mnist = torchvision.datasets.MNIST(root, train, download=False)
        cifar = torchvision.datasets.CIFAR10(root, train, download=False)

        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
        self.img_num_list = img_num_list
        self.gen_imbalanced_data(img_num_list, mnist, cifar)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        if self.train:
            img_max = 5000
            img_num_per_cls = []
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls = [1000] * 10
        return img_num_per_cls

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (img_mnist, img_cifar), target = self.data[index], self.targets[index]

        if self.train:
            img1_cifar = self.transform[0][0](img_cifar)
            img1_mnist = self.transform[1][0](img_mnist)
            img1 = torch.cat((img1_mnist, img1_cifar), 1)

            img2_cifar = self.transform[0][1](img_cifar)
            img2_mnist = self.transform[1][1](img_mnist)
            img2 = torch.cat((img2_mnist, img2_cifar), 1)
            return [img1, img2], target
        else:
            img_mnist = self.transform_val_mnist(img_mnist)
            img_cifar = self.transform_val_cifar(img_cifar)
            img = torch.cat((img_mnist, img_cifar), 1)
            return img, target

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def get_num_classes(self):
        return self.cls_num

    def get_cls_num_list(self):
        return self.img_num_list

    def gen_imbalanced_data(self, img_num_per_cls, mnist, cifar):
        new_data = []
        new_targets = []
        targets_mnist = mnist.targets.numpy()
        targets_cifar = np.array(cifar.targets, dtype=np.int64)
        classes = list(range(self.cls_num))
        self.num_per_cls = img_num_per_cls
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # self.num_per_cls_dict[the_class] = the_img_num
            idx_mnist = np.where(targets_mnist == the_class)[0]
            idx_cifar = np.where(targets_cifar == the_class)[0]
            if idx_mnist.shape[0] < the_img_num:
                idx_mnist = np.concatenate((idx_mnist, idx_mnist), 0)
            np.random.shuffle(idx_mnist)
            np.random.shuffle(idx_cifar)
            seleidx_mnist = idx_mnist[:the_img_num]
            seleidx_cifar = idx_cifar[:the_img_num]
            new_data.extend([(F.to_pil_image(img_mnist.unsqueeze(0).repeat((3, 1, 1))),
                              Image.fromarray(img_cifar)) for img_mnist, img_cifar in
                             zip(mnist.data[seleidx_mnist], cifar.data[seleidx_cifar])])
            new_targets.extend([the_class] * the_img_num)
        self.data = new_data
        self.targets = new_targets
