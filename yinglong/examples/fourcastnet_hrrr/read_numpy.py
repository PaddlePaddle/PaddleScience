import numpy as np

mean = np.load("/root//ssd3/datasets/era5_tiny/stat/global_means.npy")
print(mean.shape)  # (1, 21, 1, 1)
std = np.load("/root//ssd3/datasets/era5_tiny/stat/global_stds.npy")
print(std.shape)  # (1, 21, 1, 1)


time_mean = np.load("/root//ssd3/datasets/era5_tiny/stat/time_means.npy")
print(time_mean.shape)  # (1, 21, 721, 1440)
