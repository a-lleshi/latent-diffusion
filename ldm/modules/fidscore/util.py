# This is a custom implementation of the FID score for Latent Diffusion Models created by LatentDiffusion at comp vis.
# By Abdurrahman Lleshi

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import linalg
from PIL import Image

def get_fid_score(real_images, fake_images, batch_size=50, cuda=True, dims=2048):
    """Computes the FID between two sets of images"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    model.eval()
    m1, s1 = calculate_activation_statistics(real_images, model, batch_size, dims, cuda)
    m2, s2 = calculate_activation_statistics(fake_images, model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calculate_activation_statistics(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : The images whose statistics should be calculated. Should be an instance of numpy array.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, the activations are copied to GPU using cuda
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_activations(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values of the images must lie between 0 and 255.
    -- model       : Instance of inception model
    -- batch_size  : Images are loaded in batches of this size.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, the activations are copied to GPU using cuda
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the activations of the given tensor when
       feeding inception with the query tensor.
    """
    dataloader = DataLoader(images, batch_size=batch_size)
    pred_arr = np.empty((len(images), dims))
    for i, batch in enumerate(dataloader, 0):
        if cuda:
            batch = batch.cuda()
        pred = model(batch)[0]
        pred_arr[i * batch_size:(i + 1) * batch_size] = pred.cpu().data.numpy()
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of the pool_3 layer of the
               inception net ( like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning pool3 features"""
    BLOCK_INDEX_BY_DIM = {
        64: -1,
        192: -2,
        768: -3,
        2048: -4,
    }
    def __init__(self, output_blocks, resize_input=True, normalize_input=True):
        super(InceptionV3, self).__init__()
        self.output_blocks = output_blocks
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        inception = models.inception_v3(pretrained=True, transform_input=False)
        self.Mixed_7c = nn.Sequential(*list(inception.children())[0][:-3])
        self.Mixed_8a = nn.Sequential(*list(inception.children())[0][-3:-2])
        self.Mixed_8b = nn.Sequential(*list(inception.children())[0][-2:-1])
        self.Mixed_8c = nn.Sequential(*list(inception.children())[0][-1:])
        self.avgpool = inception.avgpool
        self.dropout = inception.dropout
        self.fc = inception.fc
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.clone()
        if self.normalize_input:
            x /= 255.0
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.Mixed_7c(x)
        x = self.Mixed_8a(x)
        x = self.Mixed_8b(x)
        x = self.Mixed_8c(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return [x]