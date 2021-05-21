from skimage.measure import label, regionprops


def connection(label_image, background=0, return_num=False, connectivity=None):
    return label(label_image, background, return_num, connectivity)


def get_regions(label_image, intensity_image=None,
                cache=True, extra_properties=None):
    return regionprops(label_image, intensity_image=intensity_image,
                       cache=cache, extra_properties=extra_properties)
