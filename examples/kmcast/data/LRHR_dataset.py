import paddle
import xarray as xr


def transform_minus_one_to_one(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


class LRHRDataset(paddle.io.Dataset):
    def __init__(self, dataset_opt, split="train", data_len=-1):
        val_samples = dataset_opt.val_samples
        self.split = split
        ds_gfs = xr.open_dataset(dataset_opt.ds_gfs_path).isel(
            uv=slice(0, 2), lat=slice(0, 192), lon=slice(0, 256)
        )
        ds_wrf = xr.open_dataset(dataset_opt.ds_wrf_path).isel(
            lat=slice(0, 192), lon=slice(0, 256)
        )
        self.lr_data = ds_gfs.GFS13110.values.transpose((1, 0, 2, 3))
        self.hr_data = ds_wrf.WRF13110.values.transpose((1, 0, 2, 3))
        self.lr_min, self.lr_max = 0, 1
        self.hr_min, self.hr_max = (
            self.hr_data[:, 1, :, :] * (39.674255 + 36.528397)
            - 36.528397
            - (self.lr_data[:, 1, :, :] * (34.886944 + 34.006493) - 34.006493)
        ).min(), (
            self.hr_data[:, 1, :, :] * (39.674255 + 36.528397)
            - 36.528397
            - (self.lr_data[:, 1, :, :] * (34.886944 + 34.006493) - 34.006493)
        ).max()
        self.hr_data = transform_minus_one_to_one(
            self.hr_data[:, 1, :, :] * (39.674255 + 36.528397)
            - 36.528397
            - (self.lr_data[:, 1, :, :] * (34.886944 + 34.006493) - 34.006493),
            self.hr_min,
            self.hr_max,
        )
        self.lat, self.lon = ds_gfs.lat.values, ds_gfs.lon.values
        if split == "train":
            self.data_len = 13110 - val_samples
            if data_len > 0:
                self.data_len = min(data_len, self.data_len)
            self.lr_data = self.lr_data[: self.data_len]
            self.hr_data = self.hr_data[: self.data_len]
        else:
            self.data_len = val_samples
            if data_len > 0:
                self.data_len = min(data_len, self.data_len)
            self.lr_data = self.lr_data[-val_samples:]
            self.hr_data = self.hr_data[-val_samples:]
        if len(tuple(self.lr_data.shape)) == 4:
            self.hr_data = self.hr_data[:, None, :, :]

    def __getitem__(self, index):
        lr_data = self.lr_data[index]
        hr_data = self.hr_data[index]
        return {"LR": lr_data, "HR": hr_data, "Index": index}

    def __len__(self):
        return self.data_len
