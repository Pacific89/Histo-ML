
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
import h5py
import numpy as np
from scipy.signal import find_peaks
from PIL import Image
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import openslide
import random
import gc
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

def heatmap_to_binary(path, original_path, name):

    image = skimage.io.imread(fname=path)
    orig_image = skimage.io.imread(fname=original_path)



    img_r = image[:,:,0]
    img_b = image[:,:,2]

    val = filters.threshold_otsu(img_b)
    hist, bins_center = exposure.histogram(img_b)

    adapted_val = val - val*0.3

    binary_mask = img_b > adapted_val

    arr_ = Image.fromarray(binary_mask)
    image_ = Image.fromarray(image, 'RGB')
    orig_image_ = Image.fromarray(orig_image, 'RGB')

    print(val, adapted_val)

    images = [orig_image_, image_, arr_]
    width = orig_image.shape[1]
    height = int(orig_image.shape[0]) #+ orig_image.shape[1]*0.3)

    total_width = int(width*3)
    max_height = height

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        print(im)
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('{0}.jpg'.format(name))


    plt.figure(figsize=(30, 14))
    ax1 = plt.subplot(131)
    ax1.title.set_text('Original')
    plt.imshow(orig_image)
    plt.axis('off')

    ax2 = plt.subplot(132)
    ax2.title.set_text('CLAM-Heatmap')
    plt.imshow(image)
    plt.axis('off')

    ax3 = plt.subplot(133)
    ax3.title.set_text('Binary Mask')
    plt.imshow(binary_mask, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')

    plt.tight_layout()
    plt.show()

def get_patch(coords, wsi_path, patch_size):

    wsi = openslide.OpenSlide(wsi_path)
    patch = wsi.read_region(tuple(coords), level=0, size=(patch_size, patch_size))

    # plt.figure()
    # plt.imshow(patch)
    # plt.show()

    return patch

def create_patch_dict(coords, name, patch_size):
    patch_dict = {}
    count = 0
    for coord in coords:
        patch = get_patch(coord, wsi_path, patch_size)
        id_ = name + "_" + str(count)
        patch_dict[id_] = patch
        count += 1

    return patch_dict

def export_patches(coords, attentions, name, base_path, patch_size):

    count = 0
    patch_folder = os.path.join(base_path, "patches")
    dpi = 80
    figsize = patch_size / float(dpi), patch_size / float(dpi)

    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)

    wsi = openslide.OpenSlide(wsi_path)
    
    for ind in tqdm(range(len(coords))):

    # for ind, coord in tqdm(enumerate(coords)):
        # patch = get_patch(coord, wsi_path, patch_size)
        coord = coords[ind]
        patch = wsi.read_region(tuple(coord), level=0, size=(patch_size, patch_size))

        att = attentions[ind]
        id_ = name + "_" + str(count) + "_score_" + str(att)
        count += 1

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(patch)
        # plt.suptitle(id_)
        # plt.show()
        path = os.path.join(patch_folder, id_)
        plt.savefig(path + ".png", dpi=dpi)

        plt.cla() 
        plt.clf() 
        plt.close('all')
        gc.collect

    del wsi

def extract_patches(path, wsi_path, name, base_path=None, patch_size=256, min_val=-15, num_patches=25, get_patch_dict=False, shuffle=True, store_single_patches=True):
    
    
    with h5py.File(path, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        attention_data = np.array(f["attention_scores"]).flatten()
        coordinates = np.array(f["coords"])

    sorted_indices = np.argsort(attention_data)

    sorted_attentions = attention_data[sorted_indices]
    sorted_coords = coordinates[sorted_indices]

    highest_attentions = sorted_attentions[sorted_attentions > min_val]
    lowest_attentions = sorted_attentions[sorted_attentions < min_val]
    coords_highest = sorted_coords[sorted_attentions > min_val]
    coords_lowest = sorted_coords[sorted_attentions < min_val]

    # print("Highest attentions: ", highest_attentions[-25:])
    # print("Lowest attentions: ", lowest_attentions[:25])

    # print("Coordinates Highest Attentions: ", coords_highest[-25:])
    # print("Coordinates Lowest Attentions: ", coords_lowest[:25])

    num_patches = min(num_patches, len(coords_highest))
    indices = np.arange(num_patches)

    if shuffle:
        np.random.shuffle(indices)


    coords = coords_highest[indices]
    attentions = highest_attentions[indices]

    if get_patch_dict:
        patch_dict = create_patch_dict(coords, name, patch_size)
        return patch_dict
    
    if store_single_patches:
        export_patches(coords, attentions, name, base_path, patch_size)

def attention_histogram(path, heatmap_path, name, pp):

    image = skimage.io.imread(fname=heatmap_path)

    with h5py.File(path, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        attention_data = np.array(f["attention_scores"]).flatten()
        coordinates = np.array(f["coords"])
    
    # print(attention_data)
    bins, values = np.histogram(attention_data, bins=30)
    otsu_val = filters.threshold_otsu(attention_data)

    peaks, _ = find_peaks(bins, height=50, prominence=2)

    try:
        local_maxima = values[peaks]

        local_hist = bins[peaks[0]:peaks[-1]]
        local_min = np.argmin(local_hist)

        min_bin = local_hist[local_min]
        min_val = values[peaks[0]+local_min]

    except:
        min_val = otsu_val

    fig = plt.figure(figsize=(12,9))
    plt.suptitle(name)
    ax1 = plt.subplot(211)
    ax1.title.set_text('CLAM-Heatmap')
    plt.axis('off')
    plt.imshow(image)
    ax1 = plt.subplot(212)
    ax1.title.set_text('Histogram Attention Scores')
    plt.hist(attention_data, bins=30)
    plt.plot(values[:-1], bins)
    plt.scatter(local_maxima, _["peak_heights"], marker="x", c="green")
    plt.vlines(min_val, 0, max(bins), colors="purple", label="local min")
    plt.vlines(otsu_val, 0, max(bins), colors="k", label="Otsu")
    plt.legend()
    # plt.tight_layout()
    plt.show()

    pp.savefig(fig)


    


if __name__ == "__main__":

    # heatmap_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/production/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667_0.5_roi_0_blur_0_rs_1_bc_0_a_0.4_l_-1_bi_0_-1.0.jpg"
    # orig_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/production/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667_orig_2.jpg"
    
    # patch_h5_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/raw/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667/DigitalSlide_A1M_11S_1_20190127143432667_0.5_roi_False.h5"
    # patch_h5_block = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/raw/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667/DigitalSlide_A1M_11S_1_20190127143432667_blockmap.h5"

    parent_path = "/media/user/easystore/HRD-Subset-IV"
    # parent_path = "/home/simon/philipp/HRD-Subset-V"

    base_path = "/media/user/easystore"
    # base_path = "/home/simon/philipp"

    # pdf_name = 'heat_checks.pdf'
    # if os.path.isfile(pdf_name):
    #     print("PDF Exists... Aborting")

    #     sys.exit()
    # pp = PdfPages(pdf_name)
    # min_vals = [-15,-10,-5,0,5]
    num_patches = 200
    min_val = 0
    patch_dict = {}
    for root, dirs, files in os.walk(parent_path):
        # print(root)
        if "data" in root:
            if len(files) == 1:
                wsi_path = os.path.join(root, files[0])

        for subfolder in dirs:
            if subfolder == "production":
                img_dir = os.path.join(root, "production/heat/Unspecified")
                
                for img in os.listdir(img_dir):
                    if "orig_2" in img:
                        orig_path = os.path.join(img_dir, img)
                        # print("Original: ", orig_path)
                    else:
                        heat_path = os.path.join(img_dir, img)
                        # print("Heat: ", heat_path)

                name = root.split("IV/")[1].split("/")[0]
                print(name)
                # print(heat_path)

                # heatmap_to_binary(heat_path, orig_path, name)

            elif subfolder == "raw":
                for r, d, f in os.walk(os.path.join(root, "raw/heat/Unspecified/{0}".format(name))):
                    h5_files = [x for x in f if x.endswith("h5")]
                    
                    try:
                        blockmap = [x for x in f if x.endswith("blockmap.h5")][0]
                    except:
                        print("No Blockmap Found")
                        blockmap = []

                    if isinstance(blockmap, str):
                        blockmap_path = os.path.join(r, blockmap)
                        # print(blockmap_path)

                        # attention_histogram(blockmap_path, heat_path, name, pp)
                        patch_sub_dict = extract_patches(blockmap_path, wsi_path, name, min_val=min_val, num_patches=num_patches, get_patch_dict=True, store_single_patches=False)

                        # extract_patches(blockmap_path, wsi_path, name, base_path, min_val=min_val, num_patches=num_patches)

                        patch_dict.update(patch_sub_dict)
    

    # pp = PdfPages("patches_test_thresh_{0}.pdf".format(min_val))

    count = 1
    items = list(patch_dict.items())
    random.shuffle(items)
    shuffled_patch_dict = dict(items)

    for p in shuffled_patch_dict:

        if count == 1:
            fig = plt.figure(figsize=(16, 10))

        pos = int(count%25+1)
        plt.subplot(5,5,pos)
        plt.axis('off')
        plt.imshow(patch_dict[p])

        if count == 25:
            pp.savefig(fig)
            plt.show()
            plt.close(fig)
            fig = plt.figure(figsize=(16, 10))
            count = 0
        
        count += 1

    # pp.close()

    # attention_histogram(patch_h5_block, heatmap_path)


