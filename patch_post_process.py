import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import gc
import matplotlib
import json
import openslide
import h5py
from pathlib import Path
from tqdm import tqdm
# matplotlib.use('Agg')


class Patch_pdf():

    def __init__(self, parent_path, json_file=None):

        self.json_file = json_file
        self.parent_path = parent_path
        self.project_name = self.parent_path.split("/")[-1]
        self.save_path = Path(self.parent_path).parent
        
        if json_file:
            self.set_hover_data()

            self.sorted_files = self.get_sorted_files()

            print("Sorted File Names: ")
            [print(f) for f in self.sorted_files]
            print("Hover Legend: ", self.json_file)

        else:
            self.get_clam_paths()

    def set_hover_data(self):

        with open(self.json_file) as info_json:
            info_data = json.load(info_json)


        self.hover_labels = [info_data[x][0] for x in info_data]
        self.hover_colors = [info_data[x][1] for x in info_data]


    def get_sorted_files(self):

        for root, dirs, files in os.walk(self.parent_path):
            # print(root)
            if "data" in root:
                print("Data folder: ", root)
                data_folder = root
                # print(patch_files[:10])

            for subfolder in dirs:
                if subfolder == "overlay":
                    overlay_folder = os.path.join(root, subfolder)
                    patch_files = os.listdir(overlay_folder)
                    print("Overlay: ", overlay_folder)


        wsi_file_names = self.get_file_names(patch_files)
        sorted_files = sorted(wsi_file_names, key=len)[::-1]

        return sorted_files

    def export_patches(self, coords, attentions, name, wsi_path, patch_size):

        count = 0
        patch_folder = os.path.join(self.save_path, "patches")
        patch_folder = os.path.join(patch_folder, name)
        patch_folder = os.path.join(patch_folder, "data")
        dpi = 80
        figsize = patch_size / float(dpi), patch_size / float(dpi)

        if not os.path.isdir(patch_folder):
            os.makedirs(patch_folder)

        wsi = openslide.OpenSlide(wsi_path)
        
        for ind in tqdm(range(len(coords))):
            coord = np.array(coords[ind]/2).astype(int)
            patch = wsi.read_region(tuple(coord), level=1, size=(patch_size, patch_size))

            att = attentions[ind]
            id_ = name + "_" + str(count) + "_score_" + str(att)
            count += 1

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.imshow(patch)
            # plt.suptitle(id_)
            plt.show()
            path = os.path.join(patch_folder, id_)
            plt.savefig(path + ".png", dpi=dpi)

            plt.cla() 
            plt.clf() 
            plt.close('all')
            gc.collect

        del wsi

    def extract_patches_png(self, min_val=-15, max_patches=0, patch_size=256, shuffle=False):


        total_patches = 0
        wsi_count = 0
        for wsi in tqdm(self.wsi_dict):

            if wsi_count > 20:
                break
            # print(wsi)
            wsi_path = self.wsi_dict[wsi][0]
            blockmap_path = self.wsi_dict[wsi][1]

            coords, attentions, max_coords = self.get_attentions_and_coords(blockmap_path, wsi_path, wsi, min_val)

            if max_patches > 0:
                num_patches = min(max_patches, len(coords))

            else:
                num_patches = len(coords)

            indices = np.arange(len(coords))[::-1]

            if shuffle:
                np.random.shuffle(indices)

            indices = indices[:num_patches]
            coords = coords[indices]
            attentions = attentions[indices]

            reduced_coords = len(coords)
            print("Number Patches: {0} | {1}% | Max {2}".format(reduced_coords, reduced_coords*100/max_coords, max_coords))
            if len(attentions) > 0:
                print("Score Range: [{0} - {1}]".format(attentions[-1], attentions[0]))
            else:
                print("Scores {0}".format(attentions))

            total_patches += reduced_coords
            # if get_patch_dict:
            #     patch_dict = create_patch_dict(coords, name, patch_size)
            #     return patch_dict
            
            self.export_patches(coords, attentions, wsi, wsi_path, patch_size)

            wsi_count += 1


        print("Total {0} patches extracted from {1} WSI files".format(total_patches, len(self.wsi_dict)))

    def get_attentions_and_coords(self, blockmap_path, wsi_path, name, min_val, patch_size=256, num_patches=25):
    
    
        with h5py.File(blockmap_path, "r") as f:
            # List all groups
            # print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]

            # Get the data
            attention_data = np.array(f["attention_scores"]).flatten()
            coordinates = np.array(f["coords"])

        max_coords = len(coordinates)
        sorted_indices = np.argsort(attention_data)

        sorted_attentions = attention_data[sorted_indices]
        sorted_coords = coordinates[sorted_indices]

        highest_attentions = sorted_attentions[sorted_attentions > min_val]
        lowest_attentions = sorted_attentions[sorted_attentions < min_val]
        coords_highest = sorted_coords[sorted_attentions > min_val]
        coords_lowest = sorted_coords[sorted_attentions < min_val]

        return coords_highest, highest_attentions, max_coords


    def get_clam_paths(self):
        wsi_paths = []
        blockmap_paths = []
        wsi_names = []
        print("Getting Files for Project: ", self.project_name)

        for root, dirs, files in os.walk(self.parent_path):
            # print(root)
            if "data" in root:
                if len(files) == 1:
                    wsi_path = os.path.join(root, files[0])
                    # print(wsi_path)
                    wsi_paths.append(wsi_path)

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

                    name = root.split(self.project_name + "/")[1].split("/")[0]
                    wsi_names.append(name)
                    # print(name)

                elif subfolder == "raw":
                    blockmap_file = os.path.join(root, "raw/heat/Unspecified/{0}/{0}_blockmap.h5".format(name))
                    # print("Block:", blockmap_file)
                    if os.path.isfile(blockmap_file):
                        blockmap_paths.append(blockmap_file)
                    else:
                        blockmap_paths.append("missing")


        print("Total Files: ", len(wsi_names))
        wsi_dict = {}
        missing_blockmaps = []
        for ind, block in enumerate(blockmap_paths):
            if not block == "missing":
                # print("Missing Blockmap: ", wsi_names[ind])
                wsi_dict.update({wsi_names[ind]: [wsi_paths[ind], blockmap_paths[ind]]})
            else:
                missing_blockmaps.append(wsi_names[ind])

        if len(wsi_names) == len(list(wsi_dict.keys())):
            print("All Blockmaps Available for project {0}".format(self.project_name))
        else:
            print("Blockmaps Missing for:", missing_blockmaps)

        print("Proceeding with {0} files".format(len(list(wsi_dict.keys()))))

        self.wsi_dict = wsi_dict

    def process_files(self):

        # patches per file saved if = 0 all patches available are used
        num_patches_per_file = 16

        for wsi_name in self.sorted_files:
            print("Name: ", wsi_name)

            patch_inds = np.array([int(ind) for ind, p in enumerate(patch_files) if wsi_name in p])
            patches = np.array(patch_files)[patch_inds]
            patch_files = np.delete(np.array(patch_files),patch_inds)

            create_pdf(patches, data_folder, overlay_folder, wsi_name)


    def create_pdf(self, patch_files, data_folder, overlay_folder, wsi_name, num_patches=0):

        pdf_name = '{0}.pdf'.format(wsi_name)
        pp = PdfPages(pdf_name)

        rows = 4
        grid_size = rows**2
        count = 1


        patch_files_process = patch_files
        np.random.shuffle(patch_files_process)
        
        if num_patches > 0:
            patches = patch_files_process[:num_patches]
        else:
            patches = patch_files_process

        for patch in patches:

            if count == 1:
                fig = plt.figure(figsize=(16, 10))
                indices_orig = np.arange(1,17,2)
                indices_overlay = np.arange(2,17,2)


            orig_path = os.path.join(data_folder, patch)
            overlay_path = os.path.join(overlay_folder, patch)

            img = cv2.imread(orig_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay = cv2.imread(overlay_path)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            pos_original = indices_orig[count-1]

            pos_overlay = indices_overlay[count-1]
            print(pos_original, pos_overlay)
            print(patch)
            plt.subplot(rows,rows,pos_original)
            plt.axis('off')
            plt.imshow(img)

            plt.subplot(rows,rows, pos_overlay)
            plt.axis('off')
            plt.imshow(overlay)

            del overlay
            del img

            if count == 1:

                for x in range(len(self.labels)):
                    plt.plot(0,0, label=self.labels[x], color=np.array(self.colors[x])/255)
                plt.legend(bbox_to_anchor=(0.7, 1.5),
                                    loc='upper left', borderaxespad=0. , ncol=2)

            if count == int(grid_size/2):
                pp.savefig(fig)
                # plt.show()
                fig.clf()
                plt.close(fig)
                del fig
                count = 0
            
            count += 1

        if count > 1:
            pp.savefig(fig)
            plt.show()
            fig.clf()
            plt.close()
            gc.collect()
            del fig


        pp.close()

        plt.cla() 
        plt.clf() 
        plt.close('all')
        gc.collect
        del pp

    def get_file_names(self, patch_files):

        s = "_"
        file_names = [s.join(x.split("_score")[0].split("_")[:-1]) for x in patch_files]
        unique_files = list(set(file_names))
        
        print("Patches from {0} WSI found".format(len(unique_files)))

        return unique_files

if __name__ == "__main__":

    # heatmap_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/production/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667_0.5_roi_0_blur_0_rs_1_bc_0_a_0.4_l_-1_bi_0_-1.0.jpg"
    # orig_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/production/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667_orig_2.jpg"
    
    # patch_h5_path = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/raw/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667/DigitalSlide_A1M_11S_1_20190127143432667_0.5_roi_False.h5"
    # patch_h5_block = "/home/user/Documents/Master/data/DigitalSlide_A1M_11S_1_20190127143432667/results/dfc30663e68f4ab7baa2e6c6efa3eb9a/raw/heat/Unspecified/DigitalSlide_A1M_11S_1_20190127143432667/DigitalSlide_A1M_11S_1_20190127143432667_blockmap.h5"

    # parent_path = "/media/user/easystore/HRD-Subset-I"
    parent_path = "/home/simon/philipp/HRD-Subset-II"
    # parent_path = "/media/user/easystore/patches_0_thresh_26wsi"

    json_file = "/home/user/Documents/Master/hover_net/type_info.json"
    
    patch_to_pdf = Patch_pdf(parent_path)
    patch_to_pdf.extract_patches_png(max_patches=0, min_val=-20)



                