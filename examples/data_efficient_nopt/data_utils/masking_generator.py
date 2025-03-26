import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        assert mask_ratio < 1 and mask_ratio >= 0
        self.mask_ratio = mask_ratio
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        if self.mask_ratio > 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                    np.ones(self.num_masks_per_frame),
                ]
            )
        elif self.mask_ratio == 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        assert mask_ratio < 1 and mask_ratio >= 0
        self.height, self.width = input_size
        self.mask_ratio = mask_ratio
        self.frames = 1
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        if self.mask_ratio > 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_masks_per_frame),
                    np.ones(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        else:
            mask_per_frame = np.hstack(
                [
                    np.ones(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask.astype(np.float16)
