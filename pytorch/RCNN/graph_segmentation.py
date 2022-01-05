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
    ax.set_title(f"Segmentation image (N unique regions={len(img[:, :, 3])})")
    plt.show()

def extract_regions(img: np.array) -> {}:
    segments = img[:, :, 3]
    regions = {}
    for y, label in enumerate(segments): # Iterate over horizontal axis
        for x, l in enumerate(label): # Iterate over vertical axis
            if l not in regions:
                regions[l] = {
                    "min_x": np.inf,
                    "min_y": np.inf,
                    "max_x": 0,
                    "max_y": 0,
                    "labels": l
                }

            # Bounding boxes for segments
            if regions[l]["min_x"] > x:
                regions[l]["min_x"] = x
            if regions[l]["min_y"] > y:
                regions[l]["min_y"] = y
            if regions[l]["max_x"] < x:
                regions[l]["max_x"] = x
            if regions[l]["max_y"] < y:
                regions[l]["max_y"] = y

    for key in regions.keys():
        r = regions[key]
        if (r["min_x"] == r["max_x"]) or (r["min_y"] == r["max_y"]):
            del regions[key]
    return regions


def plt_rectangle(plt, label, x1, y1, x2, y2, color="yellow", alpha=0.5):
    linewidth = 3
    if type(label) == list:
        linewidth = len(label)  * 3 + 2
        label = ""
        
    plt.text(x1, y1, label, fontsize=20, backgroundcolor=color, alpha=alpha)
    plt.plot([x1,x1], [y1,y2], linewidth=linewidth, color=color, alpha=alpha)
    plt.plot([x2,x2], [y1,y2], linewidth=linewidth, color=color, alpha=alpha)
    plt.plot([x1,x2], [y1,y1], linewidth=linewidth, color=color, alpha=alpha)
    plt.plot([x1,x2], [y2,y2], linewidth=linewidth, color=color, alpha=alpha)

# figsize = (20,20)
# plt.figure(figsize=figsize)    
# plt.imshow(img[:,:,:3])
# for item in regions.values():
#     x1 = item["min_x"]
#     y1 = item["min_y"]
#     x2 = item["max_x"]
#     y2 = item["max_y"]
#     label = item["labels"]
#     plt_rectangle(plt, label, x1, y1, x2, y2, color=np.random.choice(list(sns.xkcd_rgb.values())))
# plt.show()

# plt.figure(figsize=figsize)    
# plt.imshow(img[:,:,3])
# for item in regions.values():
#     x1 = item["min_x"]
#     y1 = item["min_y"]
#     x2 = item["max_x"]
#     y2 = item["max_y"]
#     label = item["labels"]
#     plt_rectangle(plt, label, x1, y1, x2, y2, color=np.random.choice(list(sns.xkcd_rgb.values())))
# plt.show()


def calc_texture_grad(img: np.array):
    ret = np.zeros(img.shape[:3])
    for channel in (0, 1, 2):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(
            img[:, :, channel], 8, 1.
        )
    return ret


def calc_hsv(img):
    hsv = skimage.color.rgb2hsv(img[:,:,:3])
    return hsv


def plot_image_with_min_max(img, title):
    img = img[:, :, :3]
    plt.imshow(img)
    plt.title("{} min={:5.3f}, max={:5.3f}".format(
        title,
        np.min(img),
        np.max(img))
    )
    plt.show()
