import paddle


def inv_transform_minus_one_to_one(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


def tensor2rawdata(tensor, min_val, max_val, norm_min_max=(0, 1)):
    """Not Support auto convert *.clamp_, please judge whether it is Pytorch API and convert by yourself"""
    tensor = tensor.squeeze().clip_(*norm_min_max)
    tensor = inv_transform_minus_one_to_one(tensor, min_val, max_val)
    return tensor


def save_img(img_turple, latlon_turple, img_path):
    sr_img, hr_img, lr_img = img_turple
    lat, lon = latlon_turple
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes[0, 0].set_title("SR (Model Output)")
    axes[0, 1].set_title("HR (Ground Truth)")
    axes[0, 2].set_title("LR (Bilinear)")
    vmin = min(sr_img.min(), hr_img.min(), lr_img.min())
    vmax = max(sr_img.max(), hr_img.max(), lr_img.max())
    cmap = plt.cm.RdYlBu_r
    for i in range(3):
        axes[i, 0].pcolormesh(lon, lat, sr_img[i], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[i, 0].set_ylabel(f"Sample {i + 1}")
        axes[i, 1].pcolormesh(lon, lat, hr_img[i], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[i, 2].pcolormesh(lon, lat, lr_img[i, 1], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close()


def calculate_rmse_sum(img1, img2):
    mse = paddle.mean(x=(img1 - img2) ** 2, axis=(-2, -1))
    return paddle.sqrt(x=mse).sum()
