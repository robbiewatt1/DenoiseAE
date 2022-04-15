import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


"""
Dataset and data augmentation classes for edge radiation images.
"""


class NoiseTransform:
    """
    Augments data by adding Poisson noise and Poisson background to image
    """
    def __init__(self, max_photons, background_photons, image_norm,
                 image_shape):
        """
        :param max_photons: Average max photons on image
        :param background_photons: Average background photons
        :param image_norm: Image normalizing value
        :param image_shape: Shape of image
        """
        self.max_photons = max_photons
        self.background_photons = background_photons
        self.image_norm = image_norm
        self.background_array = background_photons * torch.ones(image_shape)

    def __call__(self, image):
        """
        :param image: Input image
        :return: Augmented image
        """
        image = torch.clip(image, min=0)
        return torch.poisson(image * self.max_photons / self.image_norm)\
            + torch.poisson(self.background_array)


class EdgeRadDataset(Dataset):
    """
    Dataset for de-nosing edge radiation images. Provides noisy input image
    and down-sampled noiseless image.
    """
    def __init__(self, data_path, max_photons, background_photons,
                 down_sample=4):
        """
        :param data_path: Path to h5 file containing data
        :param max_photons: Max average number of photons
        :param background_photons: Average background photon number
        :param down_sample: Down sampling factor for output image.
        """
        self.max_photons = max_photons
        self.background_photons = background_photons

        file = h5py.File(data_path, 'r')
        self.images = file['Images']
        self.image_norm = np.max(self.images[0])
        image_shape = self.images.shape[1:3]
        self.transform = NoiseTransform(max_photons, background_photons,
                                        self.image_norm, image_shape)
        self.pool = torch.nn.AvgPool2d(down_sample)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        Apply transform and convert to pytorch tensors.
        """
        image = torch.tensor(self.images[idx], dtype=torch.float)[None, :, :]
        image_in = self.transform(image) / (self.max_photons
                                            + self.background_photons)
        image_out = self.pool(image) / self.image_norm
        return image_in, image_out
