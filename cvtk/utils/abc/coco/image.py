from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def count_image_size(img_dir, **kw):
    data = defaultdict(int)

    for img_path in Path(img_dir).glob("**/*"):
        try:
            if img_path.suffix in IMG_EXTENSIONS:
                img = Image.open(img_path)
                data[img.size] += 1
        except Exception as e:
            print(f"{img_path.name} - {e}")

    data = [(k[0], k[1], v) for k, v in data.items()]
    data = sorted(data, key=lambda x: x[0])

    return [(f"{a}x{b}", c) for a, b, c in data]
