import torch
import torch.nn.functional as F
from ddpm import Diffusion
from model import UNet
from dataset.helpers import *
from util import show, set_seed, CLASS_LABELS

def sample_images_with_classes(ddpm_model, unet_model, num_samples, class_combinations, device):
    """
    Sample images with specified class combinations.
    
    Args:
    - ddpm_model: The DDPM model for sampling.
    - unet_model: The U-Net model used by the DDPM model.
    - num_samples: Number of samples to generate.
    - class_combinations: List of class combinations for each sample, e.g., [[1, 1, 0, 0, 0]] for class 0 and 1.
    - device: The device (CPU or CUDA) to use for computations.
    
    Returns:
    - imgs: A list of normalized images generated by the model.
    """
    # Ensure unet_model is in evaluation mode
    unet_model.eval()
    
    # Convert class combinations to a tensor
    y = torch.tensor(class_combinations, device=device).float()
    
    # Generate samples using the DDPM model
    x_new = ddpm_model.p_sample_loop(unet_model, num_samples, y=y)
    
    # Normalize and convert tensors to images
    imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in x_new]
    
    return imgs

# Example usage
if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize DDPM and UNet models
    ddpm_cFg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cFg', device=device)
    unet_ddpm_cFg = UNet(num_classes=5, device=device)
    unet_ddpm_cFg.eval()
    unet_ddpm_cFg.to(device)
    unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))
    
    # Define class combinations (e.g., [1, 1, 0, 0, 0] for classes 0 and 1)
    class_combinations = [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1,0,1,0,0], [1,0,0,1,0], [1,0,0,0,1]]  # Modify as needed
    # class_combinations = class_combinations + [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]  # Modify as needed
    
    # Sample images
    imgs = sample_images_with_classes(ddpm_cFg, unet_ddpm_cFg, len(class_combinations), class_combinations, device)
    
    # Display sampled images
    show(imgs, fig_titles=[f"Classes: {'&'.join([CLASS_LABELS[i] for i, value in enumerate(comb) if value])}" for comb in class_combinations], title='classifier FREE guidance', save_path='assets/cFg_samples.png')
