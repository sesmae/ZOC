from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
from torchvision.datasets import ImageFolder
import glob


class cifar10_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cifar10_isolated_class, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar10 = CIFAR10(root='./data', train=False, download=True)

        class_mask = np.array(cifar10.targets) == cifar10.class_to_idx[class_label]
        self.data = cifar10.data[class_mask]
        self.targets = np.array(cifar10.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cifar10_single_isolated_class_loader():
    splits = [[0, 1, 9, 7, 3, 2],
              [0, 2, 4, 3, 7, 5],
              [5, 1, 9, 8, 7, 0],
              [5, 7, 1, 8, 4, 6],
              [8, 1, 5, 3, 4, 6]]
    loaders_dict = {}
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #cifar10_labels = ['automobile', 'ship']
    for label in cifar10_labels:
       dataset = cifar10_isolated_class(label)
       loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
       loaders_dict[label] = loader
    return loaders_dict


class cifar100_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cifar100_isolated_class, self).__init__()
        superclass_list =  [['aquatic mammals',	'beaver', 'dolphin', 'otter', 'seal', 'whale'],
                           ['fish',	'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                           ['flowers',	'orchid','poppy', 'rose', 'sunflower', 'tulip'],
                           ['food container',	'bottle', 'bowl', 'can', 'cup', 'plate'],
                           ['fruit and vegetables',	'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                           ['household electrical devices',	'clock', 'keyboard', 'lamp', 'telephone', 'television'],
                           ['household furniture', 'bed', 'chair', 'couch', 'table', 'wardrobe'],
                           ['insects',	'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                           ['large carnivores',	'bear', 'leopard', 'lion', 'tiger', 'wolf'],
                           ['large man-made outdoor things', 'bridge', 'castle', 'house', 'road', 'skyscraper'],
                           ['large natural outdoor scenes',	'cloud', 'forest', 'mountain', 'plain', 'sea'],
                           ['large omnivores and herbivores', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                           ['medium-sized mammals',	'fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                           ['non-insect invertebrates',	'crab', 'lobster', 'snail', 'spider', 'worm'],
                           ['people',	'baby', 'boy', 'girl', 'man', 'woman'],
                           ['reptiles',	'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                           ['small mammals',	'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                           ['trees', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                           ['vehicles',	'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                           ['large vehicles', 	'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        fine_to_coarse_dict = {}
        for superclass in superclass_list:
            fine_to_coarse_dict.update(dict.fromkeys(superclass[1:], superclass[0]))
        self.fine_label = class_label
        self.coarse_label = fine_to_coarse_dict[class_label]

        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar100 = CIFAR100(root='./data', train=False, download=True)

        self.coarse_label = fine_to_coarse_dict[class_label]

        class_mask = np.array(cifar100.targets) == cifar100.class_to_idx[class_label]
        self.data = cifar100.data[class_mask]
        self.targets = np.array(cifar100.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cifar100_single_isolated_class_loader():
    loaders_dict = {}
    cifar100_labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                           'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                           'orchid','poppy', 'rose', 'sunflower', 'tulip',
                           'bottle', 'bowl', 'can', 'cup', 'plate',
                           'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
                           'clock', 'keyboard', 'lamp', 'telephone', 'television',
                           'bed', 'chair', 'couch', 'table', 'wardrobe',
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                           'bear', 'leopard', 'lion', 'tiger', 'wolf',
                           'bridge', 'castle', 'house', 'road', 'skyscraper',
                           'cloud', 'forest', 'mountain', 'plain', 'sea',
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                           'crab', 'lobster', 'snail', 'spider', 'worm',
                           'baby', 'boy', 'girl', 'man', 'woman',
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                           'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
                           'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
                           'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    for label in cifar100_labels:
       dataset = cifar100_isolated_class(label)
       loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
       loaders_dict[label] = loader
    return loaders_dict


class cifarplus():
    def __init__(self, class_list):
        self.class_list = class_list
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),  # 224 for vit, 288 for res50x4
            CenterCrop(224),  # 224 for vit, 288 for res50x4
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        if len(self.class_list) == 4:
           cifar10 = CIFAR10(root='./data', train=False, download=True, transform=self.transform)
           inds = [i for i in range(len(cifar10.targets)) if cifar10.targets[i] in self.class_list]
           self.data = cifar10.data[inds]
           self.targets = np.array(cifar10.targets)[inds].tolist()
        else:
            cifar100 = CIFAR100(root='./data', train=False, download=True, transform=self.transform)
            inds = [i for i in range(len(cifar100.targets)) if cifar100.targets[i] in self.class_list]
            self.data = cifar100.data[inds]
            self.targets = np.array(cifar100.targets)[inds].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = self.transform(Image.fromarray(self.data[index]).convert('RGB'))
        return img, self.targets[index]


def cifarplus_loader():
    in_list = [0, 1, 8, 9]
    out_dict = {'plus50': [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 15, 19, 21, 31, 38,
                       34, 63, 64, 66, 75, 26, 45, 77, 79, 99, 2, 11, 35, 46, 98, 27, 29, 44, 78, 93, 36, 50, 65, 74, 80],
                       'plus10-1':[43, 36, 24, 18, 80, 98, 30, 93, 78, 3],
                       'plus10-2':[74, 91, 98, 79, 50, 66, 24, 26, 6, 42],
                       'plus10-3':[79, 63, 36, 4, 29, 55, 75, 46, 72, 38],
                       'plus10-4':[95, 93, 26, 43, 36, 27, 18, 30, 64, 32],
                       'plus10-5':[88, 18, 19, 24, 65, 50, 4, 93, 35, 46]}

    in_dataset = cifarplus(in_list)
    in_loader = DataLoader(dataset=in_dataset, batch_size=1, num_workers=4, shuffle=False)
    out_loaders = {}
    for key in out_dict.keys():
        out_dataset = cifarplus(out_dict[key])
        out_loaders[key] = DataLoader(dataset=out_dataset, batch_size=1, num_workers=4, shuffle=False)
    return in_loader, out_loaders


def tinyimage_semantic_spit_generator():
   tinyimage_splits = [
       [192, 112, 145, 107, 91, 180, 144, 193, 10, 125, 186, 28, 72, 124, 54, 77, 157, 169, 104, 166],
       [156, 157, 167, 175, 153, 11, 147, 0, 199, 171, 132, 60, 87, 190, 101, 111, 193, 71, 131, 192],
       [28, 15, 103, 33, 90, 167, 61, 13, 124, 159, 49, 12, 54, 78, 82, 107, 80, 25, 140, 46],
       [128, 132, 123, 72, 154, 35, 86, 10, 188, 28, 85, 89, 91, 82, 116, 65, 96, 41, 134, 25],
       [102, 79, 47, 106, 59, 93, 145, 10, 62, 175, 76, 183, 48, 130, 38, 186, 44, 8, 29, 26]]  # CAC splits
   dataset = ImageFolder(root='./data/tiny-imagenet-200/val')
   a=dataset.class_to_idx
   reverse_a = {v:k for k,v in a.items()}
   semantic_splits = [[],[],[],[],[]]
   for i, split in enumerate(tinyimage_splits):
       wnid_split = []
       for idx in split:
           wnid_split.append(reverse_a[idx])
       all = list(dataset.class_to_idx.keys())
       seen = wnid_split
       unseen = list(set(all)-set(seen))
       seen.extend(unseen)
       f = open('./dataloaders/imagenet_id_to_label.txt', 'r')
       imagenet_id_idx_semantic = f.readlines()

       for id in seen:
           for line in imagenet_id_idx_semantic:
               if id == line[:-1].split(' ')[0]:
                   semantic_label = line[:-1].split(' ')[2]
                   semantic_splits[i].append(semantic_label)
                   break
   return semantic_splits


class tinyimage_isolated_class(Dataset):
    def __init__(self, label, mappings):
        assert label, 'a semantic label should be specified'
        super(tinyimage_isolated_class, self).__init__()
        path = './data/tiny-imagenet-200/val/'
        #path = '/Users/Sepid/data/tiny-imagenet-200/val/'
        self.image_paths = glob.glob(os.path.join(path, mappings[label], '*.JPEG'))
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x


def tinyimage_single_isolated_class_loader():
    semantic_splits = tinyimage_semantic_spit_generator()
    f = open('./dataloaders/tinyimagenet_labels_to_ids.txt', 'r')
    #f = open('../tinyimagenet_ids_to_label.txt', 'r')
    tinyimg_label2folder = f.readlines()
    mappings_dict = {}
    for line in tinyimg_label2folder:
        label, class_id = line[:-1].split(' ')[0], line[:-1].split(' ')[1]
        mappings_dict[label] = class_id

    loaders_dict = {}
    for semantic_label in mappings_dict.keys():
        dataset = tinyimage_isolated_class(semantic_label, mappings_dict)
        loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
        loaders_dict[semantic_label] = loader
    return semantic_splits, loaders_dict


if __name__ == '__main__':
   splits = [[0, 1, 9, 7, 3, 2],
              [0, 2, 4, 3, 7, 5],
              [5, 1, 9, 8, 7, 0],
              [5, 7, 1, 8, 4, 6],
              [8, 1, 5, 3, 4, 6]]
   dset = CIFAR100(root='/Users/Sepid/data')
   idx2cls = {v:k for k,v in dset.class_to_idx.items()}
   for i, split in enumerate(splits):
       print('split{}'.format(i))
       ls=[]
       for idx in split:
           ls.append(idx2cls[idx])
       print(set(dset.class_to_idx.keys())-set(ls))