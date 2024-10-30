import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def crop_3Dimage(seg, image_sa, center, size, affine_matrix=None):
    """Crop a 3D image using a bounding box centred at (c0, c1, c2) with specified size (size0, size1, size2)"""
    c0, c1, c2 = center
    size0, size1, size2 = size
    S_seg = tuple(seg.shape)
    S0, S1, S2 = S_seg[0], S_seg[1], S_seg[2]
    r0, r1, r2 = int(size0 / 2), int(size1 / 2), int(size2 / 2)
    start0, end0 = c0 - r0, c0 + r0
    start1, end1 = c1 - r1, c1 + r1
    start2, end2 = c2 - r2, c2 + r2
    start0_, end0_ = max(start0, 0), min(end0, S0)
    start1_, end1_ = max(start1, 0), min(end1, S1)
    start2_, end2_ = max(start2, 0), min(end2, S2)
    crop = seg[start0_:end0_, start1_:end1_, start2_:end2_]
    crop_img = image_sa[start0_:end0_, start1_:end1_, start2_:end2_]
    if crop_img.ndim == 3:
        crop_img = np.pad(
            crop_img,
            (
                (start0_ - start0, end0 - end0_),
                (start1_ - start1, end1 - end1_),
                (start2_ - start2, end2 - end2_),
            ),
            "constant",
        )
        crop = np.pad(
            crop,
            (
                (start0_ - start0, end0 - end0_),
                (start1_ - start1, end1 - end1_),
                (start2_ - start2, end2 - end2_),
            ),
            "constant",
        )
    else:
        crop_img = np.pad(
            crop_img,
            (
                (start0_ - start0, end0 - end0_),
                (start1_ - start1, end1 - end1_),
                (start2_ - start2, end2 - end2_),
                (0, 0),
            ),
            "constant",
        )
        crop = np.pad(
            crop,
            (
                (start0_ - start0, end0 - end0_),
                (start1_ - start1, end1 - end1_),
                (start2_ - start2, end2 - end2_),
                (0, 0),
            ),
            "constant",
        )
    if affine_matrix is None:
        return crop, crop_img
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        affine_matrix[0:3, -1] = R.dot(np.array([c0 - r0, c1 - r1, c2 - r2])) + b
        return crop, crop_img, affine_matrix


def np_categorical_dice(pred, truth, k):
    """Dice overlap metric for label k"""
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def np_mean_dice(pred, truth):
    """Dice mean metric"""
    dsc = []
    for k in np.unique(truth)[1:]:
        dsc.append(np_categorical_dice(pred, truth, k))
    return np.mean(dsc)


def combine_labels(input_paths, pad=-1, seed=None):
    def get_most_popular(count_map):
        return max(count_map, key=count_map.get)

    def is_equivocal(count_map):
        return len(set(count_map.values())) > 1

    def decide_on_tie(count_map, rng):
        max_count = max(count_map.values())
        tied_labels = [
            label for label, count in count_map.items() if count == max_count
        ]
        return rng.choice(tied_labels)

    def calculate_counts(input_paths, output_shape):
        counts = [{} for _ in range(np.prod(output_shape))]
        for input_path in input_paths:
            input_image = nib.load(input_path).get_fdata().astype(np.int32)
            contended_voxel_indices = np.where(
                np.logical_and(
                    output != input_image,
                    np.logical_or(output > pad, input_image > pad),
                )
            )
            idx = np.ravel_multi_index(contended_voxel_indices, output_shape)
            labels = input_image[contended_voxel_indices]
            _, counts_per_label = np.unique(idx, return_counts=True)
            for idx, label, count in zip(idx, labels, counts_per_label):
                counts[idx][label] = counts[idx].get(label, 0) + count
        return counts

    output_image = nib.load(input_paths[0])
    output_data = output_image.get_fdata().astype(np.uint8)
    output_shape = tuple(output_data.shape)
    unanimous_mask = np.ones(output_shape, dtype=np.uint8)
    output = output_data.copy()
    counts = calculate_counts(input_paths, output_shape)
    contended_voxel_indices = np.where(unanimous_mask == 0)
    idx = np.ravel_multi_index(contended_voxel_indices, output_shape)
    for idx, (z, y, x) in zip(idx, np.transpose(contended_voxel_indices)):
        output[z, y, x] = get_most_popular(counts[idx])
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    equivocal_voxel_indices = np.where(unanimous_mask == 0)
    idx = np.ravel_multi_index(equivocal_voxel_indices, output_shape)
    unique_indices, counts_per_voxel = np.unique(idx, return_counts=True)
    for idx, (z, y, x) in zip(unique_indices, np.transpose(equivocal_voxel_indices)):
        if is_equivocal(counts[idx]):
            output[z, y, x] = decide_on_tie(counts[idx], rng)
    return output


def threshold_image(data, threshold=130):
    # Perform thresholding using NumPy operations
    thresholded_data = data.copy()
    thresholded_data[data <= threshold] = 0
    thresholded_data[(data > threshold)] = 1
    return thresholded_data


def blur_image(data, sigma):
    # Apply Gaussian blurring to the data using scipy.ndimage.gaussian_filter
    blurred_data = gaussian_filter(data, sigma=sigma)
    return blurred_data


def binarize_image(data, lower_threshold=4, upper_threshold=4, binary_value=255):
    # Perform binarization using NumPy operations
    binarized_data = np.zeros_like(data)
    binarized_data[(data >= lower_threshold) & (data <= upper_threshold)] = binary_value
    return binarized_data


def padding(imageA, imageB, threshold, padding, invert=False):
    # Create a mask for positions that require padding
    if invert:
        mask = imageB != threshold
    else:
        mask = imageB == threshold

    # Update 'imageA' using the mask and padding value
    imageA[mask] = padding
    return imageA


def refineFusionResults(data, alfa):
    data = np.round(data)

    hrt = threshold_image(blur_image(binarize_image(data, 1, 4), alfa), 130)
    rvendo = threshold_image(blur_image(binarize_image(data, 4, 4), alfa), 130)
    lvepi = threshold_image(blur_image(binarize_image(data, 1, 2), alfa), 115)
    lvendo = threshold_image(blur_image(binarize_image(data, 1, 1), alfa), 130)

    hrt = padding(hrt, hrt, 1, 4)
    rvendo = padding(hrt, rvendo, 1, 4)
    lvepi = padding(rvendo, lvepi, 1, 2)
    data_final = padding(lvepi, lvendo, 1, 1)
    return data_final
