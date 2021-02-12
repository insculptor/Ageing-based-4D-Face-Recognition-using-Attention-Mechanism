import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from skimage.io import imread
import trimesh
from PIL import Image
from torchvision import transforms
import Imports.Helper.binvox_rw as binvox_rw
import glob

###

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data


class BosphorousDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])

    def __getitem__(self, index):

        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_file = model_3d_file[:-7] + ".png"
        #with open(self.root + model_3d_file, "rb") as f:
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        #print(volume.shape)
        #plotFromVoxels(volume)
        #with open(self.root + model_2d_file, "rb") as g:
        #image = np.array(imread(self.root + model_2d_file))
        image = Image.open(self.root + model_2d_file)
        image = np.asarray(self.p(image))
        return (torch.FloatTensor(image), torch.FloatTensor(volume) )

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])