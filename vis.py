# debug_visualizer.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

def feat_to_rgb(feat, H, W, name):
    """
    将 [N, C] 的特征图用 PCA 降到 [N, 3]，然后 reshape 成 [H, W, 3] 并保存。
    """
    pca = PCA(n_components=3)
    feat_np = feat.detach().cpu().numpy()
    feat_rgb = pca.fit_transform(feat_np)
    
    # 标准化到0~1
    feat_rgb -= feat_rgb.min()
    feat_rgb /= feat_rgb.max() + 1e-6
    feat_rgb = feat_rgb.reshape(H, W, 3)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(feat_rgb)
    plt.axis("off")
    plt.title(name)
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"))
    plt.close()
    
    # Also return torch tensor for TensorBoard
    return torch.from_numpy(feat_rgb).permute(2, 0, 1)  # [3, H, W]

SAVE_DIR = "./vis_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

def imsave(name, arr, cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    plt.title(name)
    plt.axis("off")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"))
    plt.close()

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1)

def l2_dist(a, b):
    return ((a - b)**2).mean(dim=-1)

def visualize():
    data = torch.load("debug_outputs.pt")
    disp = data["disp_pred"][0].squeeze(-1)  # [N]
    w_r = data["w_r"][0]  # [N, C]
    w_l = data["w_l"][0]
    f_r = data["f_r"][0]
    f_l = data["f_l"][0]

    # 恢复图像尺寸（假设图像为正方形，或可手动指定 H, W）
    
    H , W = 32, 96
    disp_map = disp.view(H, W).numpy()
    

    # 可视化disp
    imsave("disparity_pred", disp_map, cmap="jet")



    # Cosine similarity heatmap
    sim_wl_fl = cosine_sim(w_l, f_l).view(H, W).numpy()
    sim_wr_fr = cosine_sim(w_r, f_r).view(H, W).numpy()
    imsave("cosine(w_l, f_l)", sim_wl_fl, cmap="plasma")
    imsave("cosine(w_r, f_r)", sim_wr_fr, cmap="plasma")

    # L2 distance heatmap
    l2_wl_fl = l2_dist(w_l, f_l).view(H, W).numpy()
    l2_wr_fr = l2_dist(w_r, f_r).view(H, W).numpy()
    imsave("l2(w_l, f_l)", l2_wl_fl, cmap="hot")
    imsave("l2(w_r, f_r)", l2_wr_fr, cmap="hot")


    # 可视化特征图（PCA压缩成RGB）
    feat_imgs = {}
    feat_imgs["w_l"] = feat_to_rgb(w_l, H, W, "w_l (PCA)")
    feat_imgs["w_r"] = feat_to_rgb(w_r, H, W, "w_r (PCA)")
    feat_imgs["f_l"] = feat_to_rgb(f_l, H, W, "f_l (PCA)")
    feat_imgs["f_r"] = feat_to_rgb(f_r, H, W, "f_r (PCA)")

    # TensorBoard logging
    writer = SummaryWriter("./logs")
    writer.add_image("disp_pred", torch.from_numpy(disp_map).unsqueeze(0), 0, dataformats='CHW')
    writer.add_image("cosine(w_l, f_l)", torch.from_numpy(sim_wl_fl).unsqueeze(0), 0)
    writer.add_image("cosine(w_r, f_r)", torch.from_numpy(sim_wr_fr).unsqueeze(0), 0)
    writer.add_image("l2(w_l, f_l)", torch.from_numpy(l2_wl_fl).unsqueeze(0), 0)
    writer.add_image("l2(w_r, f_r)", torch.from_numpy(l2_wr_fr).unsqueeze(0), 0)
    for name, img in feat_imgs.items():
        writer.add_image(name, img, 0)
    writer.close()

    print(f"✅ Visualization complete. Images saved to: {SAVE_DIR}")
    print("📉 Run `tensorboard --logdir=./logs` to launch TensorBoard.")

if __name__ == "__main__":
    visualize()