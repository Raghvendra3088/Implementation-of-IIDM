import matplotlib.pyplot as plt
import os

def save_plots(pred, gt, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gt.cpu()[0,0], cmap='Greens')
    ax[0].set_title("Ground Truth")
    ax[1].imshow(pred.cpu()[0,0], cmap='Greens')
    ax[1].set_title("Prediction")
    ax[2].imshow((gt - pred).cpu()[0,0], cmap='coolwarm')
    ax[2].set_title("Error Map")
    plt.savefig(f"{save_dir}/comparison.png")
