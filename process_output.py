import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.WholeSlideImage import WholeSlideImage
import os
import random
import matplotlib
from PIL import Image
from skimage import filters
import cv2
from skimage.measure import label, regionprops
from matplotlib import cm
import matplotlib.colors as mcolors
from PIL import ImageColor
from tqdm import tqdm
import gc
import threading


class SuperPatcher():

    def __init__(self, wsi_name, slide_path, mask_path):

        self.wsi_name = wsi_name
        self.slide_path = slide_path
        self.mask_path = mask_path

        if os.path.isdir("/media/user/easystore"):
            self.base_path = "/media/user/easystore/clam_heat_superpatches"
        elif os.path.isdir("/home/simon/philipp"):
            self.base_path = "/home/simon/philipp/clam_heat_superpatches"

        self.save_path = os.path.join(self.base_path, self.wsi_name)
        self.patch_path = os.path.join(self.save_path, "patches")
        self.data_path = os.path.join(self.patch_path, "data")

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.isdir(self.patch_path):
            os.makedirs(self.patch_path)

        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)


        self.wsi_object = WholeSlideImage(self.slide_path)
        self.wsi_object.initSegmentation(self.mask_path)
        self.wsi_openslide = self.wsi_object.getOpenSlide()



    def get_scores(self, blockmap_path):

        df = pd.DataFrame(columns=["coords", "attention_scores"])
        with h5py.File(blockmap_path, "r") as f:
            # List all groups
            keys = list(f.keys())

            # Get the data
            self.coords = f[keys[1]][:]
            self.scores = f[keys[0]][:]

        # hist = np.histogram(self.scores)
        # steepest_descend = np.argmin(np.diff(hist[0]))
        # thresh_med = np.median(self.scores)
        # thresh_des = hist[1][steepest_descend]
        # otsu = filters.threshold_otsu(self.scores)
        # max_y = max(hist[0])
        # print("Median: ", thresh_med)

        # print("Steepest Desc. at: ", thresh_des)
        # print("Using Otsu Threshold: ", otsu)

        # plt.figure()
        # plt.hist(scores, bins=100)
        # plt.vlines(thresh_med,0,max_y, color='g', label='Median')
        # plt.vlines(thresh_des,0,max_y, color='k', label='Deriv.')
        # plt.vlines(otsu,0,max_y, color='purple', label='otsu')
        # plt.legend()
        # plt.show()



    def redraw_heatmap(self, vis_level=-1):


        cmap = matplotlib.colors.ListedColormap(['White', 'Blue'])
        if vis_level < 0:
            vis_level = self.wsi_openslide.get_best_level_for_downsample(32)


        
        region_size = self.wsi_object.level_dim[vis_level]
        downsample = self.wsi_object.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        contours_tissue = self.wsi_object.scaleContourDim(self.wsi_object.contours_tissue, scale)
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)



        # heatmap_var = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, segment=False, use_holes=False, blur=True, binarize=True, cmap="jet", blank_canvas=False, thresh=thresh, convert_to_percentiles=False)
        # heatmap = drawHeatmap(scores, coords, slide_path)

        # heatmap_low = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, segment=False, use_holes=False, blur=True, binarize=True, cmap="jet", blank_canvas=False, thresh=0.5, convert_to_percentiles=True)
        # heatmap_low_att = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, segment=True, use_holes=False, blur=False, binarize=True, cmap=cmap, blank_canvas=True, thresh=0.6, convert_to_percentiles=True, alpha=0.8)
        heatmap_high_att = self.wsi_object.visHeatmap(scores=self.scores, coords=self.coords, vis_level=vis_level, segment=True, use_holes=False, blur=False, binarize=True, cmap=cmap, blank_canvas=True, thresh=0.8, convert_to_percentiles=True, alpha=0.8)
        heatmap_orig = self.wsi_object.visHeatmap(scores=self.scores, coords=self.coords, vis_level=vis_level, segment=False, use_holes=False, blur=True, binarize=True, cmap="jet", blank_canvas=False, thresh=0.6, convert_to_percentiles=True)

        # heatmap_border_att = np.array(heatmap_low_att) - np.array(heatmap_high_att)
        # print(type(heatmap_high_att))
        # heatmap_border_att = cv2.cvtColor(np.array(heatmap_border_att), cv2.COLOR_BGR2RGB)
        heatmap_high_att = cv2.cvtColor(np.array(heatmap_high_att), cv2.COLOR_BGR2RGB)

        heatmap_orig = cv2.cvtColor(cv2.cvtColor(np.array(heatmap_orig), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)

        heatmap_orig, bbox_dict = self.draw_rois(heatmap_orig, heatmap_high_att, region_size)


        for super_patch in bbox_dict:
            bbox = bbox_dict[super_patch]
            self.get_zero_level_region(bbox,super_patch, downsample)
            self.save_patches_(bbox, super_patch, downsample)

        # for idx in range(len(contours_tissue)):
        #     cv2.drawContours(image=heatmap_high_att, contours=contours_tissue, contourIdx=idx, color=(0, 10, 100), thickness=12)

        # plt.figure()
        # plt.imshow(heatmap_low)


        plt.figure()
        plt.imshow(heatmap_orig)
        plt.imsave(os.path.join(self.save_path, "{0}_heatmap.jpg".format(self.wsi_name)),heatmap_orig)
        # plt.show()

    def get_zero_level_region(self, bbox, super_patch, downsample, shift=0):
        # bbox_keys = list(bbox_dict.keys())
        # bbox = np.array(bbox_dict[bbox_keys[0]])

        if shift > 0:
            bbox[0] = bbox[0] + bbox[0]*shift
            bbox[1] = bbox[1] - bbox[1]*shift

        scale_x = downsample[0]
        scale_y = downsample[1]

        color = bbox["color"]
        bbox = bbox["bbox"]

        bbox_top_left = (int(bbox[0][0]*scale_x), int(bbox[0][1]*scale_y))
        bbox_bot_right = (int(bbox[1][0]*scale_x), int(bbox[1][1]*scale_y))

        size = np.array(bbox_bot_right) - np.array(bbox_top_left)

        print("Unscaled: ", bbox)
        print("Rescaled: ", bbox_top_left, bbox_bot_right)
        print("Size: ", size)

        region = self.wsi_openslide.read_region(bbox_top_left, 0, size)
        region.save(os.path.join(self.save_path, "{0}_superpatch_{1}_pil.png".format(self.wsi_name, super_patch)))

        # dpi=1200
        # figsize = tuple(size / float(dpi))

        # fig = plt.figure(figsize=figsize, dpi=dpi)
        # ax = fig.add_axes([0, 0, 1, 1])

        # ax.imshow(region)

        # col = np.array(color)/255

        # for child in ax.get_children():
        #     if isinstance(child, matplotlib.spines.Spine):
        #         child.set_color(col)
        #         child.set_linewidth(4)

        # # ax.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        # # plt.show()

        # fig.canvas.draw()

        # # convert canvas to image
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        #         sep='')
        # img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # plt.imshow(img)
        # plt.imsave(os.path.join(self.save_path, "{0}_superpatch_{1}.jpg".format(self.wsi_name, super_patch)),img)

        # plt.show()
        print("done")

    def export_patches(self, filtered_coords, patch_size, super_patch):

        dpi = 80
        figsize = patch_size / float(dpi), patch_size / float(dpi)
        threads = []

        for ind in tqdm(range(len(filtered_coords))):
            coord = filtered_coords[ind]
            patch = self.wsi_openslide.read_region(tuple(coord), level=0, size=(patch_size, patch_size))
            id_ = self.wsi_name + "_" + "superpatch_{0}_x_{1}_y_{2}".format(super_patch, coord[0], coord[1])
            path = os.path.join(self.data_path, id_)



            t = threading.Thread(target=self.save_patch, args=[patch, path])
            t.start()
            t.join()
            # threads.append(t)

            # if (ind+1)%40:
        # for thread in threads:
            # thread.join()
        # threads = []

    def save_patch(self, patch, path):

        patch.save(path + ".png")


    def save_patches_(self, bbox, super_patch, downsample):
        patch_size = 256
        scale_x = downsample[0]
        scale_y = downsample[1]

        bbox = bbox["bbox"]
        bbox_top_left = (int(bbox[0][0]*scale_x), int(bbox[0][1]*scale_y))
        bbox_bot_right = (int(bbox[1][0]*scale_x), int(bbox[1][1]*scale_y))

        filtered_coords = self.coords[(self.coords[:,0] > bbox_top_left[0]) & (self.coords[:,0] < bbox_bot_right[0]) & (self.coords[:,1] > bbox_top_left[1]) & (self.coords[:,1] < bbox_bot_right[1])]
        
        self.export_patches(filtered_coords, patch_size, super_patch)


    def draw_rois(self, heatmap_draw, heatmap_reference, size, roi_num = 5, thresh_area=100000, get_largest=True):
        """Use heatmap_reference to generate label image and calculate regionprops.
        Take roi_num largest areas and draw them to heatmap_draw object

        Parameters
        ----------
        heatmap_draw : numpy array
            heatmap from clam visHeatmap (could be different thresholds)
        heatmap_reference : numpy array
            heatmap from clam visHeatmap (could be different thresholds)
        roi_num : int
            number of rois to draw (largest areas)
        """
        
        label_img = label(heatmap_reference[:,:,2])
        props = regionprops(label_img)

        print(len(props))

        # plt.figure()
        # plt.imshow(label_img)
        # plt.show()

        props = [p for p in props if not p.label == 1]
        areas = [p.area for p in props]
        sorted_ind = np.argsort(areas)[::-1]
        bbox_dict = {}
        # cmap = cm.get_cmap('viridis', roi_num)
        colors = list(mcolors.TABLEAU_COLORS.values())

        c = 0
        roi_count = 1
        s_patch_size = 300

        if get_largest:
            for index in sorted_ind[1:roi_num+1]:
                prop = props[index]
                bbox = prop.bbox

                centroid = prop.centroid
                pts1_ = np.array(np.array(centroid) - s_patch_size).astype(int)
                pts2_ = np.array(np.array(centroid) + s_patch_size).astype(int)

                pts1 = (pts1_[1], pts1_[0])
                pts2 = (pts2_[1], pts2_[0])

                col = ImageColor.getcolor(colors[roi_count], "RGB")

                bbox_dict.update({roi_count : {"bbox" : (pts1, pts2), "color" : col} })

                cv2.rectangle(heatmap_draw, pts1, pts2,
                        col, thickness=6)
                
                roi_count += 1

        # else:
        
        #     while True:
        #         prop = props[sorted_ind[c]]
        #         print("Area: ", prop.area)
        #         if prop.area < thresh_area:

        #             pts1 = (prop.bbox[1], prop.bbox[0])
        #             pts2 = (prop.bbox[3], prop.bbox[2])
        #             centroid = prop.centroid

        #             # pts1, pts2 = get_squares(pts1, pts2, centroid, size)
                    
        #             col = ImageColor.getcolor(colors[roi_count], "RGB")

        #             bbox_dict.update({roi_count : {"bbox" : (pts1, pts2), "color" : col} })

        #             cv2.rectangle(heatmap_draw, pts1, pts2,
        #                     col, thickness=6)
                
        #             roi_count += 1
                        
        #         c +=1
                
        #         if roi_count == roi_num+1:
        #             break

        print(bbox_dict)
        return heatmap_draw, bbox_dict



def get_coords_from_name(superpatch, files, path, save_path, patch_size=256):

    refstring = "superpatch_{0}".format(superpatch)
    overlay_files = [f for f in files if refstring in f]
    x_coords = [f.split("x_")[1].split("_")[0] for f in overlay_files]
    y_coords = [f.split("y_")[1].split(".png")[0] for f in overlay_files]

    coords = np.array(list(zip(x_coords, y_coords))).astype(int)
    
    min_x = min(coords[:,0])
    min_y = min(coords[:,1])

    # size = tuple(np.max(coords, axis=0) - np.array([min_x, min_y]) + np.array([patch_size, patch_size]))
    size = tuple((11000,11000))
    print(size)

    img = np.array(Image.new(size=size, mode="RGB", color=(255,255,255)))

    for f in overlay_files:
        with Image.open(os.path.join(path, f)) as im:
            patch_img = im.copy()

        print(f)
        coord_x = int(f.split("x_")[1].split("_")[0]) - min_x
        coord_y = int(f.split("y_")[1].split(".png")[0]) - min_y

        # print(coord_x, coord_y)
        patch_shape = np.array(patch_img).shape
        print(patch_shape)
        # if not patch_shape == (patch_size, patch_size, 3):
        #     continue
        print("Size: ", size)
        print("Y: {0} - {1}".format(coord_y, coord_y+patch_shape[1]))
        print("X: {0} - {1}".format(coord_x, coord_x+patch_shape[0]))

        img[coord_y:coord_y+patch_shape[1], coord_x:coord_x+patch_shape[0]] = patch_img


        # plt.figure()
        # plt.imshow(patch_img)
        # plt.show()

        del patch_img

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.imsave(os.path.join(save_path, "superpatch_{}_hover.jpg".format(superpatch)),img)

    # plt.show()

    return coords

def get_overlays():
    path = "/media/user/easystore/clam_heat_superpatches/DigitalSlide_A1M_10S_1_20190127142442923/patches/overlay/"
    
    save_path = path.split("/patches")[0]
    # path = "/home/simon/philipp/DigitalSlide_A2M_7S_1/results/f965f9b3758043e88652151d8e20ac8d/overlay"

    files = os.listdir(path)
    superpatch_nums = list(set([f.split("superpatch_")[1].split("_")[0] for f in files]))

    for superpatch in superpatch_nums:
        coords = get_coords_from_name(superpatch, files, path, save_path)

    print(files)


def get_super_patches(wsi, wsi_base_path):
    

    wsi_path = os.path.join(wsi_base_path, wsi)
    slide_path = os.path.join(wsi_path, os.path.join("data", "{0}.svs".format(wsi)))
    rel_block = ""
    blockmap_path = os.path.join(wsi_path, )
    print(slide_path)

    for root, dirs, files in os.walk(wsi_path):
        # print(root)
        if "raw/heat" in root:
            for f in files:
                if len(f) > 0 and f.endswith("blockmap.h5"):
                    blockmap_path = os.path.join(root, f)

                if len(f) > 0 and f.endswith("mask.pkl"):
                    mask_path = os.path.join(root, f)

    # slide_path = "/media/user/easystore/HRD-Subset-II/DigitalSlide_A1M_17S_1/data/DigitalSlide_A1M_17S_1.svs"
    # blockmap_path = "/media/user/easystore/HRD-Subset-II/DigitalSlide_A1M_17S_1/results/9a48ccfc50e343e79ca5d554548c5add/raw/heat/Unspecified/DigitalSlide_A1M_17S_1/DigitalSlide_A1M_17S_1_blockmap.h5"

    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                        'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0,'a_h':8.0,'max_n_holes':10}


    sp = SuperPatcher(wsi, slide_path, mask_path)
    print("SuperPatcher", sp)
    sp.get_scores(blockmap_path)
    sp.redraw_heatmap()

def start_processing():

    wsi_base_path = "/media/user/easystore/HRD-Subset-II"

    if os.path.isdir(wsi_base_path):
        wsi_paths = os.listdir(wsi_base_path)

    else:
        wsi_base_path = "/home/simon/philipp/HRD-Subset-II"
        wsi_paths = os.listdir(wsi_base_path)

    # random wsi:
    # wsi = random.choice(wsi_paths)
    # first wsi:
    # wsi = wsi_paths[0]

    for wsi in wsi_paths:
        try:
            get_super_patches(wsi, wsi_base_path)
        except:
            continue


if __name__ == "__main__":

    
    # start_processing()
    get_overlays()



