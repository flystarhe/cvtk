import random
from functools import wraps

import numpy as np

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


@clipped
def gauss_noise(image, gauss):
    image = image.astype("float32")
    return image + gauss


class GaussNoise(object):

    def __init__(self, var_limit=(10.0, 50.0), mean=0):
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")
            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                f"Expected var_limit type to be one of (int, float, tuple, list), got {type(var_limit)}"
            )

        self.mean = mean

    def __call__(self, image):
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        gauss = random_state.normal(self.mean, sigma, image.shape)
        return gauss_noise(image, gauss)


def peak(X, Y, mu1, mu2, sigma1, sigma2, p=0):
    """like matlab peaks function

    Args:
        mu1 (float): Mean ("centre") of the distribution.
        mu2 (float): Mean ("centre") of the distribution.
        sigma1 (float): Standard deviation (spread or "width") of the distribution.
        sigma2 (float): Standard deviation (spread or "width") of the distribution.
        p (float): `-1 < p < 1`, function always `p=0`.

    Returns:
        arr : ndarray

    Example1:
    ```python
    import numpy as np

    # 3D surface
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)

    X, Y = np.meshgrid(X, Y)
    Z = np.exp(-(X**2 + Y**2))
    vmin, vmax = Z.min(), Z.max()

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(figsize=[8, 8], subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(vmin - 0.01, vmax - 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    ```

    Example2:
    ```python
    import numpy as np

    Z = np.diag(range(15))
    vmin, vmax = Z.min(), Z.max()

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.matshow(Z, cmap=cm.jet, vmin=vmin, vmax=vmax)
    plt.show()
    ```
    """
    X, Y = (X - mu1) / sigma1, (Y - mu2) / sigma2
    Z = np.exp(-0.5 * (X**2 + Y**2)) / (2 * np.pi * sigma1 * sigma2)
    return Z


def peak_region(w, h, mu1, mu2, sigma1, sigma2):
    X = np.linspace(-3, 3, w) * sigma1
    Y = np.linspace(-3, 3, h) * sigma2
    X, Y = np.meshgrid(X, Y)

    Z = peak(X, Y, mu1, mu2, sigma1, sigma2)
    return Z


class PeakNoise(object):

    def __init__(self, var_limit=(1.0, 30.0)):
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")
            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                f"Expected var_limit type to be one of (int, float, tuple, list), got {type(var_limit)}"
            )

    def __call__(self, image):
        img_h, img_w, img_c = image.shape
        assert (img_h >= 64) and (img_w >= 64)

        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma1 = var ** 0.5

        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma2 = var ** 0.5

        w = np.random.randint(16, img_w // 3)
        h = np.random.randint(16, img_h // 3)
        x = np.random.randint(0, img_w - w)
        y = np.random.randint(0, img_h - h)

        noise = peak_region(w, h, 0, 0, sigma1, sigma2)
        noise = noise / noise.max() * random.uniform(30, 110)
        noise = np.stack([noise for _ in range(img_c)], axis=2)

        dtype = image.dtype
        image = image.astype("float32")
        if random.random() < 0.5:
            image[y: y + h, x: x + w] += noise
        else:
            image[y: y + h, x: x + w] -= noise

        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return np.clip(image, 0, maxval).astype(dtype)
