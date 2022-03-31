from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt

import gc


def convert(file_path):
    img = cv2.imread(file_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(type(img))

    im = Image.fromarray(img_rgb)

    im.save(os.path.join(file_path.replace("png", "jpg")))

    # plt.figure()
    # plt.imshow(img)
    # plt.axis('off')
    # plt.imsave(os.path.join(file_path.replace("png", "jpg")),img)

    # plt.cla()
    # plt.clf() 
    # plt.close('all')
    # gc.collect

    del img
    del img_rgb
    del im

    # os.remove(file_path)


if __name__ == "__main__":
    path = "/media/user/easystore/clam_heat_superpatches"

    # print(os.listdir(path))

    subfolders = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]

    for p in subfolders:
        full_path = os.path.join(path, p)
        # print(full_path)
        files = [os.path.join(full_path, f) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.endswith("png")]
        print("Files to convert: ", len(files))
        # print(files)
        for file_ in files:
            print(file_)
            convert(file_)