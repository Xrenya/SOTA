import skimage
import skimage.segmentation
import matplotlib.pyplot as plt
import numpy as np


def plot_image_segmentation(img_path: str=None):
    """This fast 2D image segmentation algorithm
    Efficient graph-based image segmentation,
    Felzenszwalb, P.F. and Huttenlocher, D.P.
    International Journal of Computer Vision, 2004

    http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf

    :param img_path:
    :return:
    """
    if img_path:
        img = skimage.io.imread(fname=img_path)
    else:
        img = skimage.data.astronaut()
    # Increasing each parameter will leave only a large area.
    scales = [100, 300, 100, 100]
    sigmas = [0.5, 0.5, 0.8, 0.5]
    min_sizes = [50, 50, 50, 200]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, (scale, sigma, min_size) in enumerate(zip(scales, sigmas, min_sizes)):
        segments = skimage.segmentation.felzenszwalb(img,
                                                     scale=scale,
                                                     sigma=sigma,
                                                     min_size=min_size)
        ax[i // 2][i % 2].imshow(skimage.segmentation.mark_boundaries(img, segments))
        ax[i // 2][i % 2].set_title(f"Scale={scale}, "
                                    f"sigma={sigma}, "
                                    f"min_size={min_size}")
    plt.show()


def image_segmentation(img_path: str,
                       scale: int=1,
                       sigma: float=0.5,
                       min_size: int=50
) -> np.array:
    """

    :param img_path: str
        Image path
    :param scale: int
        Free parameter. Higher means larger clusters in felzenszwalb segmentation.
    :param sigma: float
        Width of Gaussian kernel for felzenszwalb segmentation.
    :param min_size: int
        Minimum component size for felzenszwalb segmentation.
    :return:
    img : ndarray
        Image with region label
        Region label is stored in the 4th value of each pixel [r, g, b, region]
    """
    img = skimage.io.imread(fname=img_path)
    img_masks = skimage.segmentation.felzenszwalb(
        img,
        scale=scale,
        sigma=sigma,
        min_size=min_size
    )
    img = np.dstack([img, img_masks])
    return img


def plot_img_and_mask(img: np.array):
    fig = plt.figure(figsize=(15,30))
    ax  = fig.add_subplot(121)
    ax.imshow(img[:, :, :3])
    ax.set_title("Original image")
    ax  = fig.add_subplot(122)
    ax.imshow(img[:, :, 3])
    ax.set_title("Segmentation image")
    plt.show()
