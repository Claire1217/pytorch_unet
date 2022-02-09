
import os
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import cv2
from PIL import Image


class ADNISegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_dir: str,
                 seg_dir: str,
                 slice_axis: int,
                 number_of_slices_from_centre: int
                 ) -> None:
        super().__init__()
        self.axis = slice_axis
        self.n_slices = number_of_slices_from_centre
        self.img_dir = img_dir
        self.seg_dir = seg_dir

        self.img_files = os.listdir(img_dir)
        self.seg_files = os.listdir(seg_dir)

        assert len(self.img_files) == len(self.seg_files)

    @property
    def number_of_scans(self):
        return len(self.img_files)

    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, index):
        img_file = self.img_files[index]
        identifier = img_file.lstrip('N4_masked_').rstrip('.nii.gz')
        seg_file = f'{identifier}_spm_seg.nii.gz'

        img_arr = nib.load(os.path.join(self.img_dir, img_file)).get_fdata()
        seg_arr = nib.load(os.path.join(self.seg_dir, seg_file)).get_fdata()

        img_arr = cv2.normalize(img_arr, None, 0, 1, cv2.NORM_MINMAX)

        x, y, z = img_arr.shape

        if self.axis == 0:
            start = x//2 - self.n_slices//2
            idx = randint(start, start + self.n_slices)
            img = img_arr[idx, :, :]
            seg = seg_arr[idx, :, :, :]
        elif self.axis == 1:
            start = y//2 - self.n_slices//2
            idx = randint(start, start + self.n_slices)
            img = img_arr[:, idx, :]
            seg = seg_arr[:, idx, :, :]
        elif self.axis == 2:
            start = z//2 - self.n_slices//2
            idx = randint(start, start + self.n_slices)
            img = img_arr[:, :, idx]
            seg = seg_arr[:, :, idx, :]
        else:
            raise ValueError

        # pad to 256x256
        h, w = img.shape
        hdiff = 256 - h
        wdiff = 256 - w
        img = cv2.copyMakeBorder(
            img, hdiff//2, hdiff-hdiff//2, wdiff, wdiff-wdiff//2, cv2.BORDER_REPLICATE)
        seg = cv2.copyMakeBorder(
            seg, hdiff//2, hdiff-hdiff//2, wdiff, wdiff-wdiff//2, cv2.BORDER_REPLICATE)
        if img.shape != (256, 256):
            h, w = img.shape
            xc, yc = int(h//2), int(w//2)
            half = 128
            img = img[xc - half: xc + half, yc - half: yc + half]

        if seg.shape != (256, 256):
            h, w, _ = seg.shape
            xc, yc = int(h//2), int(w//2)
            half = 128
            seg = seg[xc - half: xc + half, yc - half: yc + half, :]

        assert seg.shape == (256, 256, 3), seg.shape

        return torch.tensor(img, dtype=torch.float32)[None, None, ...], torch.tensor(seg, dtype=torch.float32).permute(2, 0, 1)[None, ...]


if __name__ == '__main__':
    IMG_DIR = 'ADNI_T1_3T_MASKED_N4'
    SEG_DIR = 'ADNI_T1_3T_SPM_SEGS_MERGED'

    dataset = ADNISegmentationDataset(IMG_DIR, SEG_DIR, 1, 30)

    for i in range(10):
        img, seg = dataset.__getitem__(randint(0, dataset.number_of_scans))
        print(img.shape, seg.shape)
        Image.fromarray(
            cv2.normalize(img.cpu().numpy()[
                          0, 0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        ).save(f'{i}_img.png')

        Image.fromarray(
            cv2.normalize(seg.permute(0, 2, 3, 1).cpu().numpy()[
                          0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        ).save(f'{i}_seg.png')
