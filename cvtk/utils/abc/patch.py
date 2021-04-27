import shutil
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def _split(size, patch_size, overlap=128):
    if patch_size >= size:
        return [0]

    s = list(range(0, size - patch_size, patch_size - overlap))
    s.append(size - patch_size)
    return s


def _patch_image(out_dir, img_path, patch_size, overlap=128, color_mode=1):
    img = cv.imread(str(img_path), color_mode)
    img_h, img_w = img.shape[:2]

    ys = _split(img_h, patch_size, overlap)
    xs = _split(img_w, patch_size, overlap)

    cnt = 1
    stem = img_path.stem
    suffix = img_path.suffix
    for y in ys:
        for x in xs:
            out_file = out_dir / f"{stem}_{cnt:04d}{suffix}"
            sub_img = img[y: y + patch_size, x: x + patch_size]
            cv.imwrite(str(out_file), sub_img)
            cnt += 1


def patch_images(img_dir, out_dir, patch_size, overlap=128, color_mode=1):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    imgs = [img for img in img_dir.glob("**/*")
            if img.suffix in IMG_EXTENSIONS]
    for img_path in tqdm(imgs):
        _patch_image(out_dir, img_path, patch_size, overlap, color_mode)
    return str(out_dir)
