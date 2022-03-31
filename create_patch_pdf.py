import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import gc
import matplotlib
import json
matplotlib.use('Agg')


class Patch_pdf():

    def __init__(self, parent_path, json_file):

        self.json_file = json_file
        self.parent_path = parent_path
        
        self.set_hover_data()

        self.sorted_files = self.get_sorted_files()

        print("Sorted File Names: ")
        [print(f) for f in self.sorted_files]
        print("Hover Legend: ", self.json_file)

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

    parent_path = "/media/user/easystore/patches_0_thresh_26wsi"
    json_file = "/home/user/Documents/Master/hover_net/type_info.json"
    
    patch_to_pdf = Patch_pdf(parent_path, json_file)


                