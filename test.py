import openslide
import skimage
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

path = "/media/user/easystore/public_data/pat_100/BRACS_1850/data/BRACS_1850.svs"
blur_threshold = 0.1

wsi = openslide.OpenSlide(path)
img = np.array(wsi.get_thumbnail((1024,1024)))
# img = s.getImgThumb(params.get("image_work_size", "2.5x"))

img = rgb2gray(img)
img_laplace = np.abs(skimage.filters.laplace(img))
blurred_laplace = skimage.filters.gaussian(img_laplace, sigma=7)
mask = blurred_laplace <= blur_threshold

plt.figure()
plt.title("IMG GRAY")
plt.imshow(img, cmap="gray")

plt.figure()
plt.title("LAPLACE")
plt.imshow(img_laplace, cmap="gray")

plt.figure()
plt.title("BLURRED LAPLACE")
plt.imshow(mask, cmap="gray")

plt.figure()
plt.title("MASK")
plt.imshow(mask > 0)
plt.show()