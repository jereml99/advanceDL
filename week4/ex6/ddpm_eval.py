import os
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# custom imports
from ddpm import Diffusion
from model import Classifier, UNet
from dataset.helpers import *
from util import set_seed, prepare_dataloaders
from scipy.linalg import sqrtm
set_seed()

class VGG(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, features=False):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        x = self.dropout(self.flatten(feat))
        x = self.fc(x)
        if features:
            return feat
        else:
            return x
        
def get_features(model, images):
    model.eval()  
    with torch.no_grad():
        features = model(images, features=True)
    features = features.squeeze(3).squeeze(2).cpu().numpy()
    return features

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    # https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
    # HINT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    # Implement FID score
    mean_diff_sq = np.linalg.norm(mu1 - mu2, 2)
    
    # Compute sqrt of sigma1*sigma2
    covmean = sqrtm(sigma1.dot(sigma2))

    # Ensure the result is real (to avoid complex numbers due to numerical issues)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the Fréchet distance
    fid = mean_diff_sq + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


def add_random_noise(images, noise_factor=0.1):
    """
    Adds random noise to the images.
    :param images: Torch tensor of images.
    :param noise_factor: Float indicating the noise factor to be applied.
    :return: Noisy images.
    """
    if noise_factor == 0:
        return images
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    # Clipping the noisy images to maintain the original image distribution range
    noisy_images = torch.clamp(noisy_images, 0, 1)
    return noisy_images

def test_fid_with_noise(test_loader, model, device, noise_factor=0.1):
    """
    Test FID score on the same picture and a picture with random noise.
    :param test_loader: DataLoader for test dataset.
    :param model: Pre-trained VGG model for feature extraction.
    :param device: Torch device (CPU or CUDA).
    :param noise_factor: Noise factor for generating noisy images.
    """
    original_feat = []
    noisy_feat = []

    for images, _ in tqdm(test_loader):
        images = images.to(device)
        noisy_images = add_random_noise(images, noise_factor=noise_factor)
        # Extract features from original and noisy images
        original_features = get_features(model, images)
        noisy_features = get_features(model, noisy_images)

        original_feat.append(original_features)
        noisy_feat.append(noisy_features)

    # Concatenate all features from batches
    original_feat = np.concatenate(original_feat, axis=0)
    noisy_feat = np.concatenate(noisy_feat, axis=0)

    # Calculate feature statistics
    mu_original, sigma_original = feature_statistics(original_feat)
    mu_noisy, sigma_noisy = feature_statistics(noisy_feat)

    # Compute FID score between original and noisy images
    fid_score = frechet_distance(mu_original, sigma_original, mu_noisy, sigma_noisy)
    print(f'FID score between original and noisy images: {fid_score:.3f} (with noise factor: {noise_factor})')


if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################################### classifier guidance ##########################################
    ddpm_cg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg', device=device)
    classifier = Classifier(
        img_size=16, c_in=3, labels=5,
        time_dim=256,channels=32, device=device
    )
    classifier.to(device)
    classifier.eval()
    classifier.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))

    unet_ddpm = UNet(device=device)
    unet_ddpm.eval()
    unet_ddpm.to(device)
    unet_ddpm.load_state_dict(torch.load('weights/DDPM/model.pth', map_location=device))
    ddpm_cg.classifier = classifier

    ######################################### classifier-free guidance #########################################
    ddpm_cFg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cFg', device=device)
    unet_ddpm_cFg = UNet(num_classes=5, device=device)
    unet_ddpm_cFg.eval()
    unet_ddpm_cFg.to(device)
    unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))

    model = VGG()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load('weights/vgg-sprites/model.pth', map_location=device))
    dims = 256 # vgg feature dim

    _ ,_, test_loader = prepare_dataloaders(val_batch_size=100)

    vgg_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    original_feat = np.empty((len(test_loader.dataset), dims))
    generated_feat_cg = np.empty((len(test_loader.dataset), dims))
    generated_feat_cFg = np.empty((len(test_loader.dataset), dims))

    start_idx = 0
    
    test_fid_with_noise(test_loader, model, device, noise_factor=0.)
    test_fid_with_noise(test_loader, model, device, noise_factor=1)

    for images, _ in tqdm(test_loader):

        images = images.to(device)
        original = get_features(model, images)
        
        # classifier guidance
        y = torch.randint(0, 5, (images.shape[0],), device=device)
        cg_images = ddpm_cg.p_sample_loop(unet_ddpm, images.shape[0], y=y, verbose=False)
        cg_images = vgg_transform(cg_images/255.0)
        cg_features = get_features(model, cg_images)

        # classifier-free guidance
        y = F.one_hot(y, num_classes=5).float()
        cFg_images = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, images.shape[0], y=y, verbose=False)
        cFg_images = vgg_transform(cFg_images/255.0)
        cFg_features = get_features(model, cFg_images)

        # store features
        original_feat[start_idx:start_idx + original.shape[0]] = original
        generated_feat_cg[start_idx:start_idx + original.shape[0]] = cg_features
        generated_feat_cFg[start_idx:start_idx + original.shape[0]] = cFg_features

        start_idx = start_idx + original.shape[0]
    

    mu_original, sigma_original = feature_statistics(original_feat)
    mu_cg, sigma_cg = feature_statistics(generated_feat_cg)
    mu_cFg, sigma_cFg = feature_statistics(generated_feat_cFg)

    fid_cg = frechet_distance(mu_original, sigma_original, mu_cg, sigma_cg)
    fid_cFg = frechet_distance(mu_original, sigma_original, mu_cFg, sigma_cFg)
    print(f'[FID classifier guidance] {fid_cg:.3f}')
    print(f'[classifier-free guidance] {fid_cFg:.3f}')