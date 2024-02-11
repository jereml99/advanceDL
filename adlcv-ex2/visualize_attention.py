import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from einops import rearrange, repeat

# Assuming the ViT and positional_encoding_2d are defined in vit.py and imageclassification.py respectively
from vit import ViT, positional_encoding_2d
from imageclassification import prepare_dataloaders, set_seed


def denormalize(tensor, means=[0.4914, 0.4822, 0.4465], stds=[0.247, 0.243, 0.261]):
    """Denormalizes a tensor image with mean and std."""
    denormed = torch.clone(tensor)
    for channel, mean, std in zip(denormed, means, stds):
        channel.mul_(std).add_(mean)
    return denormed


def visualize_attention(original_image, attention_weights, patch_size=(4, 4), image_size=(32, 32)):
    if isinstance(original_image, torch.Tensor):
        original_image = denormalize(original_image)  # Denormalize the image
        original_image = original_image.permute(1, 2, 0).cpu().numpy()
        original_image = np.clip(original_image, 0, 1)  # Ensure the image values are valid
        original_image = (original_image * 255).astype(np.uint8)  # Scale to [0, 255]
        original_image = Image.fromarray(original_image)


    # Assuming the first dimension of attention_weights includes the CLS token
    num_patches_plus_cls = attention_weights.shape[0]
    num_patches = num_patches_plus_cls - 1  # Exclude the CLS token
    sqrt_num_patches = int(np.sqrt(num_patches))

    # Ensure the attention_weights are for patch-to-patch without CLS token
    attention_map = attention_weights[1:, 1:]  # Exclude CLS token attention weights

    # Average the attention across all patches (excluding CLS token)
    attention_map_avg = attention_map.mean(dim=0)

    # Reshape to a square matrix corresponding to the image grid of patches
    attention_map_avg = attention_map_avg.reshape(sqrt_num_patches, sqrt_num_patches)

    # Upsample to the original image size
    attention_map_resized = torch.nn.functional.interpolate(attention_map_avg.unsqueeze(0).unsqueeze(0), 
                                                            size=image_size, 
                                                            mode='bilinear', 
                                                            align_corners=False).squeeze().numpy()

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(original_image, alpha=0.9)
    ax[1].imshow(attention_map_resized, cmap='hot', alpha=0.6)
    ax[1].set_title('Attention Map Overlay')
    ax[1].axis('off')
    plt.savefig('figures/attention_map.png')



# Function to load the trained model
def load_trained_model(model_path, model_config):
    model = ViT(**model_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Model configuration must match the trained model's configuration
model_config = {
    'image_size': (32, 32),
    'channels': 3,
    'patch_size': (4, 4),
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 4,
    'pos_enc': 'learnable',
    'pool': 'cls',
    'dropout': 0.3,
    'fc_dim': None,
    'num_classes': 2,
}

# Load the trained model
model_path = 'model.pth'
model = load_trained_model(model_path, model_config)

# Prepare an example image from CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
img, label = dataset[4] 
# Convert image to batch format (B, C, H, W)
img_batch = img.unsqueeze(0)

# Get the attention weights from the model
# Note: Modify the ViT model's forward method to return attention weights as discussed
with torch.no_grad():
    _, attention_weights = model(img_batch, return_attention=True)

# Select the layer and head to visualize
layer = 0
head = 0
attention_weights_to_visualize = attention_weights[layer][head]

# Visualize the attention map
# Assuming the visualize_attention function is defined as in your snippet
visualize_attention(img, attention_weights_to_visualize.squeeze(0), patch_size=(4, 4), image_size=(32, 32))
