import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torch as th
from skimage import io
import random
from matplotlib import pyplot as plt

class LIDCDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
        else:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    #print(seqtype)
                    datapoint[seqtype] = os.path.join(root, f)
                    #print(datapoint)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = io.imread(filedict[seqtype])
            img = img / 255
            #nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(img))
        out = torch.stack(out)
        if self.test_flag:
            image = out[0]
            image = torch.unsqueeze(image, 0)
            image = torch.cat((image,image,image,image), 0) #concatenating images 4 times is not necessary for LIDC dataset, but for MRI we concatenated all of them (flair, f1, f2, pd). This is for reference! :D
            label = out[random.randint(1, 4)]
            label = torch.unsqueeze(label, 0)
  
            
            
            return (image, label, path)
        else:

            image = out[0]
            image = torch.unsqueeze(image, 0)
            image = torch.cat((image,image,image,image), 0)
            label = out[random.randint(1, 4)]
            label = torch.unsqueeze(label, 0)
            return (image, label)

    def __len__(self):
        return len(self.database)

