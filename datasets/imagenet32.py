from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class IMAGENET32(data.Dataset):

    def __init__(self, root, train=True,
                 transform=None, lbl_range = (0,1000), id_range=(1,11), debug=False):
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.lbl_range = lbl_range
        self.id_range = id_range

        
        self.data = []
        self.targets = []
        if self.train:
            for idx in range(id_range[0], id_range[1]):
                if lbl_range[1] == 1002:
                    x, y = unpickle(os.path.join(self.root, 'Imagenet32_train/train_batch_py2_') + str(idx))

                else:
                    x, y = self.loaddata(os.path.join(self.root, 'Imagenet32_train/train_data_batch_') + str(idx))
                if lbl_range[1] == 1001:
                    #dump data with protocol 2
                    with open(os.path.join(self.root, 'Imagenet32_train/train_batch_py2_') + str(idx), 'wb') as fo:
                        pickle.dump((x,y), fo, 2)

                self.data.append(x)
                self.targets.extend(y)
                print ("loaded:", idx)

        else:
            x, y = self.loaddata(os.path.join(self.root, 'Imagenet32_val/val_data'))
            self.data.append(x)
            self.targets.extend(y)


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        self.targets = [y - 1 for y in self.targets]

        
        
        if lbl_range[0] > 0 or lbl_range[1]<1000:
            _data = self.data
            _targets = self.targets
            self.data = []
            self.targets = []
            
            for i in range(_data.shape[0]):
                if _targets[i] >= lbl_range[0] and _targets[i]<lbl_range[1]:
                    self.data.append(_data[i])
                    self.targets.append(_targets[i])

            self.data = np.stack(self.data)

        
        if debug:
            _data = self.data / 255.
            avg = _data.mean(axis=(0, 1, 2))
            std = np.sqrt(((_data - avg.reshape((1, 1, 1, 3)))**2 ).mean(axis=(0, 1, 2)))
            print ("avg:", avg)
            print ("std:", std)
        
            

    def loaddata(self, path):
        d = unpickle(path)
        return d['data'], d['labels']
        

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

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

