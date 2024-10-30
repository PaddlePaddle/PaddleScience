import os
import shutil

import nibabel as nib
import numpy as np
import util.pre_process as pre_process
import util.utils as util
from scipy.ndimage import zoom
from util.image_utils import combine_labels
from util.image_utils import crop_3Dimage
from util.image_utils import np_mean_dice
from util.image_utils import refineFusionResults


def atlas_selection(
    atlas_path,
    atlas_img_type,
    atlas_nums_top,
    target_path,
    frame,
    dice_scores,
    parameter_file,
):
    # calculate the similarity between the target image and the ED/ES of atlas
    # decide to use ED or ES
    atlas_top_list = []
    for index, atlas_id in enumerate(dice_scores[:atlas_nums_top]):
        source_img = f"{atlas_path}/{atlas_id[0]}/seg_{frame}_{atlas_img_type}.nii.gz"
        target_img_atlas = f"{target_path}/tmps/seg_{frame}_{index}.nii.gz"
        affine_warped_image_path = f"{target_path}/tmps/affine_warped_source.nii.gz"
        # read atlas sa and seg, and save them in the image space

        pre_process.register_with_deepali(
            target_img,
            source_img,
            target_seg_file=target_img,
            source_seg_file=source_img,
            ffd_params_file=f"{parameter_file}_atlas_affine.yaml",
            warped_img_path=affine_warped_image_path,
        )

        pre_process.register_with_deepali(
            target_img,
            affine_warped_image_path,
            target_seg_file=target_img,
            source_seg_file=affine_warped_image_path,
            ffd_params_file=f"{parameter_file}_seg.yaml",
            warped_img_path=target_img_atlas,
        )

        target_img_atlas = f"{target_path}/tmps/seg_{frame}_{index}.nii.gz"
        seg_EDES = nib.load(
            f"{target_path}/tmps/seg_{frame}_{index}.nii.gz"
        ).get_fdata()
        seg_EDES = refineFusionResults(seg_EDES, 2)
        nib.save(nib.Nifti1Image(seg_EDES, np.eye(4)), target_img_atlas)
        atlas_top_list.append(target_img_atlas)
    return atlas_top_list


def crop_data_into_atlas_size(seg_nib, img_nib, atlas_size):
    # 0 - background, 1 - LV, 2 - MYO, 4 - RV, img_data: 4D, WHD*time_t
    # the template information 140, 140, 56
    seg_data = seg_nib.get_fdata()
    img_data = img_nib.get_fdata().squeeze()
    seg_data[seg_data == 1] = 0
    seg_data[seg_data == 3] = 1

    affine_sa = seg_nib.affine
    new_affine = affine_sa.copy()
    # align with the atlas data
    seg_data = np.flip(np.flip(seg_data, 2), 1)
    img_data = np.flip(np.flip(img_data, 2), 1)

    res_xy, res_z = seg_nib.header["pixdim"][1], seg_nib.header["pixdim"][3]
    atlas_xy, atlas_z = 1.25, 2.0

    raw_size = seg_data.shape
    # resize to atlas
    if seg_data.ndim == 3:
        seg_data = zoom(
            seg_data,
            zoom=(res_xy / atlas_xy, res_xy / atlas_xy, res_z / atlas_z),
            order=0,
        )
        img_data = zoom(
            img_data,
            zoom=(res_xy / atlas_xy, res_xy / atlas_xy, res_z / atlas_z),
            order=1,
        )
    else:
        seg_data = zoom(
            seg_data,
            zoom=(res_xy / atlas_xy, res_xy / atlas_xy, res_z / atlas_z, 1),
            order=0,
        )
        img_data = zoom(
            img_data,
            zoom=(res_xy / atlas_xy, res_xy / atlas_xy, res_z / atlas_z, 1),
            order=1,
        )
    new_affine[:3, 0] /= seg_data.shape[0] / raw_size[0]
    new_affine[:3, 1] /= seg_data.shape[1] / raw_size[1]
    new_affine[:3, 2] /= seg_data.shape[2] / raw_size[2]

    # calculate coordinates of heart center and crop
    heart_mask = (seg_data > 0).astype(np.uint8)

    c0 = np.median(np.where(heart_mask.sum(axis=-1).sum(axis=-1))[0]).astype(int)
    c1 = np.median(np.where(heart_mask.sum(axis=0).sum(axis=-1))[0]).astype(int)
    c2 = np.median(np.where(heart_mask.sum(axis=0).sum(axis=0))[0]).astype(int)

    crop_seg, crop_sa, new_affine = crop_3Dimage(
        seg_data, img_data, (c0, c1, c2), atlas_size, affine_matrix=new_affine
    )

    return crop_seg, crop_sa, new_affine


if __name__ == "__main__":
    data_dir = "./data"
    atlas_path = "./Hammersmith_myo2"
    parameter_file = "./ffd/params"
    atlas_img_type = "image_space_crop"
    atlas_nums_top = 3
    atlas_list = sorted(os.listdir(atlas_path))

    device = "gpu"
    tag = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    subject_list = sorted(os.listdir(data_dir))
    interval = len(subject_list)

    print(f"----- from dataset {tag * interval} to {(tag + 1) * interval} ------")

    for i in range(tag * interval, (tag + 1) * interval):
        subid = subject_list[i]

        print(f"----- processing {i}:{subid} ------")
        img_path = f"{data_dir}/{subid}"
        target_path = f"{data_dir}/{subid}/image_space_pipemesh"
        util.setup_dir(target_path)

        Info_path = os.path.join(img_path, "Info.cfg")
        with open(Info_path, "r") as file:
            lines = file.readlines()
        config = {}
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                config[key.strip()] = value.strip()
        ED = int(config.get("ED", 0))
        ED = str(ED).zfill(2)
        ES = int(config.get("ES", 0))
        ES = str(ES).zfill(2)

        sa_ED_path = f"{img_path}/{subid}_frame{ED}.nii.gz"
        seg_sa_ED_path = f"{img_path}/{subid}_frame{ED}_gt.nii.gz"
        sa_ED_nib = nib.load(sa_ED_path)
        seg_ED_nib = nib.load(seg_sa_ED_path)
        crop_seg_ED, crop_sa_ED, new_affine_ED = crop_data_into_atlas_size(
            seg_ED_nib, sa_ED_nib, (140, 140, 56)
        )

        sa_ES_path = f"{img_path}/{subid}_frame{ES}.nii.gz"
        seg_sa_ES_path = f"{img_path}/{subid}_frame{ES}_gt.nii.gz"
        sa_ES_nib = nib.load(sa_ES_path)
        seg_ES_nib = nib.load(seg_sa_ES_path)
        crop_seg_ES, crop_sa_ES, new_affine_ES = crop_data_into_atlas_size(
            seg_ES_nib, sa_ES_nib, (140, 140, 56)
        )

        # calculate top 3 similar atlases
        dice_scores = []
        seg_ED = crop_seg_ED
        for atlas_id in atlas_list:
            source_img = f"{atlas_path}/{atlas_id}/seg_ED_{atlas_img_type}.nii.gz"
            atlas_img = nib.load(source_img).get_fdata()
            if atlas_img.shape[2] == 56:
                # calculate the similarity between the target image and the atlas
                dice_score = np_mean_dice(seg_ED, atlas_img)
                dice_scores.append((atlas_id, dice_score))
        dice_scores.sort(key=lambda x: x[1], reverse=True)

        for frame in ["ED", "ES"]:
            # save it in the image space
            if frame == "ED":
                seg_flip_time = crop_seg_ED
                sa_flip_time = crop_sa_ED
            else:
                seg_flip_time = crop_seg_ES
                sa_flip_time = crop_sa_ES

            target_img = f"{target_path}/seg_sa_{frame}.nii.gz"
            nib.save(nib.Nifti1Image(seg_flip_time, np.eye(4)), target_img)
            util.setup_dir(f"{target_path}/tmps")

            atlas_top_list = atlas_selection(
                atlas_path,
                atlas_img_type,
                atlas_nums_top,
                target_path,
                frame,
                dice_scores,
                parameter_file,
            )
            # vote the top 3 atlases
            seg = combine_labels(atlas_top_list)

            nib.save(
                nib.Nifti1Image(sa_flip_time, np.eye(4)),
                f"{target_path}/sa_{frame}.nii.gz",
            )

            nib.save(nib.Nifti1Image(seg, np.eye(4)), target_img)

            try:
                shutil.rmtree(f"{target_path}/tmps")
            except FileNotFoundError:
                pass
