"""
The `coco` format
    `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
The `pascal_voc` format
    `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
The `albumentations` format
    is like `pascal_voc`, but normalized,
    in other words: [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
The `yolo` format
    `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
    `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
"""
