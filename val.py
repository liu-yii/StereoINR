import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random
from einops import rearrange

def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret

def load_image_as_tensor(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformation to convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales pixel values to [0, 1]
    ])
    
    # Apply the transformation
    image_tensor = transform(image)
    
    # Add a batch dimension (b, c, h, w)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def visualize_tensor(tensor, title):
    # Convert tensor to numpy array and move channel dimension to the end
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Clip values to [0, 1] range for visualization
    tensor_np = tensor_np.clip(0, 1)
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(tensor_np)
    plt.title(title)
    plt.axis('off')
    plt.show()


imgs = []
coords = []
for idx in range(1,3):
    print(f"idx: {idx}")
    # Example usage
    gt_path_L = f'demo/000{idx}_L.png'  # Replace with your image path
    gt_path_R = f'demo/000{idx}_R.png'  # Replace with your image path
    img_gt_L = load_image_as_tensor(gt_path_L)
    img_gt_R = load_image_as_tensor(gt_path_R)
    print(img_gt_L.shape)  # Should print: torch.Size([1, 3, height, width])
    b,c,h,w = img_gt_L.shape
    h_lr, w_lr = 64, 64  # Target height and width for the sampled image
    hr_coord = make_coord((img_gt_L.shape[2], img_gt_L.shape[3]))
    x0 = random.randint(0, h - h_lr)
    y0 = random.randint(0, w - w_lr)
    
    hr_coord = hr_coord[x0: x0 + h_lr, y0: y0 + w_lr,:]
    hr_coord = hr_coord.unsqueeze(0)  # Add batch and channel dimensions
    print(hr_coord.shape)  # Should print: torch.Size([1, h, w, 2])
    # Resize the left and right ground truth images to the target resolution
    img_lq_L = F.interpolate(img_gt_L, size=(h_lr, w_lr), mode='bilinear', align_corners=False)
    img_lq_R = F.interpolate(img_gt_R, size=(h_lr, w_lr), mode='bilinear', align_corners=False)
    img_gt = torch.cat([img_gt_L, img_gt_R], dim=1)  # Concatenate left and right images along channel dimension
    img_gt = img_gt[:, :, x0: x0 + h_lr, y0: y0 + w_lr]  # Crop the image to the target size
    img_lq = torch.cat([img_lq_L, img_lq_R], dim=1)  # Concatenate left and right images along channel dimension
    imgs.append(img_lq)
    coords.append(hr_coord)

img_batch = torch.cat(imgs, dim=0)  # Concatenate images along batch dimension
coord = torch.cat(coords, dim=0)  # Concatenate coordinates along batch dimension

x = torch.stack([img_batch[:, :3, :, :], img_batch[:, 3:, :, :]],dim=1) # [0, 1]
x = rearrange(x, 'b t c h w -> (b t) c h w') #[l,r,l,r]

coord_ = torch.stack([coord, coord], dim=1)
coord_ = rearrange(coord_, 'b t h w c-> (b t) h w c') #[l,r,l,r]

sample = F.grid_sample(x, coord_.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
sample_l, sample_r = torch.chunk(rearrange(sample, '(b t) c h w -> b (t c) h w', t=2), 2, dim=1)

x_l, x_r = torch.chunk(rearrange(x, '(b t) c h w -> b (t c) h w', t=2), 2, dim=1)
sample2_l = F.grid_sample(x_l, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
sample2_r = F.grid_sample(x_r, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
sample2 = torch.cat([sample2_l, sample2_r], dim=0)

visualize_tensor(x_l[0,:], "Input Tensor (Flipped Coordinates)")
visualize_tensor(x_r[0,:], "Input Tensor (Original Coordinates)")
visualize_tensor(x_l[1,:], "Input Tensor (Flipped Coordinates)")
visualize_tensor(x_r[1,:], "Input Tensor (Original Coordinates)")
visualize_tensor(sample_l[0,:], "Sampled Tensor (Flipped Coordinates)")
visualize_tensor(sample2_r[0,:], "Sampled2 Tensor (Original Coordinates)")
visualize_tensor(sample2_l[1,:], "Sampled Tensor (Flipped Coordinates)")
visualize_tensor(sample2_r[1,:], "Sampled2 Tensor (Original Coordinates)")
print("finfi")

