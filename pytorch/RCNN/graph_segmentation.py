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
    for channel in range(3):
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


def calc_hist(img, min_hist=0, max_hist=1):
    BINS = 25
    hist = np.array([])

    for channel in range(3):
        c = img[:, channel]
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (min_hist, max_hist))[0]]
        )
    hist = hist / len(img)
    return hist


def augment_regions_with_hist_info(text_grad, img, regions, hsv, tex_grad):
    for key, value in list(regions.items()):
        masked_pixels = hsv[img[:, :, 3] == key]
        regions[key]["size"] = len(masked_pixels / 4)
        regions[key]["hist_c"] = calc_hist(masked_pixels)
        regions[key]["hist_t"] = calc_hist(text_grad[img[:, :, 3] == key])
    return regions


def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"] or
           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or\
           (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or\
           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    regions_list = list(regions.items())
    neighbours = []
    for cur, a in enumerate(regions_list[:-1]):
        for b in regions_list[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours

            def sim_colour(region_1, region_2):
    return sum([
       min(a, b) for a, b in zip(region_1["hist_c"], region_2["hist_c"])
    ])


def sim_texture(region_1, region_2):
    return sum([
        min(a, b) for a, b in zip(region_1["hist_t"], region_2["hist_t"])
    ])


def sim_size(region_1, region_2, size):
    return 1. - (region_1["size"] + region_2["size"]) / size


def sim_fill(region_1, region_2, size):
    bbsize = (
        (max(region_1["max_x"], region_2["max_x"])
         - min(region_1["min_x"], region_2["min_x"]))
        * (max(region_1["max_y"], region_2["max_y"])
         - min(region_1["min_y"], region_2["min_y"]))
    )
    return 1. - (bbsize - region_1["size"] + region_2["size"]) / size


def calc_sim(region_1, region_2, size):
    return (sim_colour(region_1, region_2)
            + sim_texture(region_1, region_2)
            + sim_size(region_1, region_2, size)
            + sim_fill(region_1, region_2, size))


def caculate_simmilarity(img, neighbours, verbose=False):
    size = img.shape[0] * img.shape[1]
    s = {}
    for (ai, ar), (bi, br) in neighbours:
        s[(ai, ar)] = calc_sim(bi, br, size)
        if verbose:
            print("S[({:2.0f}, {:2.0f})]={:3.2f}".format(ai,bi,S[(ai, bi)]))
    return s


def merge_regions(region_1, region_2):
    size = region_1["size"] + region_2["size"]
    merged_region = {
        "min_x": min(region_1["min_x"], region_2["min_x"]),
        "min_y": min(region_1["min_y"], region_2["min_y"]),
        "max_x": max(region_1["max_x"], region_2["max_x"]),
        "max_y": max(region_1["max_y"], region_2["max_y"]),
        "size": size,
        "hist_c": (region_1["hist_c"] * region_1["size"]
                   + region_2["hist_c"] * region_2["size"]) / size,
        "hist_t": (region_1["hist_t"] * region_1["size"]
                   + region_2["hist_t"] * region_2["size"]) / size,
        "labels": region_1["labels"] + region_2["labels"]
    }
    return merged_region


def merge_regions_in_order(sim, regions, size, verbose=False):
    while sim != {}:
        i, j = sorted(sim.items(), key=lambda i: i[1])[-1][0]

        t = max(regions.keys()) + 1
        regions[t] = merge_regions(regions[i], regions[j])

        key_to_delete = []
        for k, v in sim.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        for key in key_to_delete:
            del sim[key]

        for k in key_to_delete:
            if k != (i, j):
                if k[0] in (i, j):
                    n = k[1]
                else:
                    n = k[0]
                sim[(t, n)] = calc_sim(regions[t], regions[n], size)

        if verbose:
            print("{} regions".format(len(regions)))

        output_regions = []
        for k, r in regions.items():
            output_regions.append({
                'rect': (
                    r['min_x'],  # min x
                    r['min_y'],  # min y
                    r['max_x'] - r['min_x'],  # width 
                    r['max_y'] - r['min_y']),  # height
                'size': r['size'],
                'labels': r['labels']
            })
        return output_regions


def get_region_proposal(img_path, min_size=500):
    img = image_segmentation(img_path, min_size=min_size)
    R = extract_regions(img)    
    tex_grad = calc_texture_grad(img)
    hsv = calc_hsv(img)
    regions = augment_regions_with_hist_info(tex_grad, img, R, hsv, tex_grad)
    del tex_grad, hsv
    neighbours = extract_neighbours(regions)
    S = calculate_similarlity(img, neighbours)
    regions = merge_regions_in_order(S, regions, size=img.shape[0] * img.shape[1])
    return regions
