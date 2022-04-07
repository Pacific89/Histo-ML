import torchvision
import torch
import PIL
from PIL import Image 
import torchvision.transforms as transforms
import os
from pathlib import Path
import numpy as np
import pandas as pd
import more_itertools
import gc
from tqdm import tqdm
import argparse
import openslide
import h5py
import warnings
from matplotlib import pyplot as plt
from wsi_loader import Whole_Slide_Bag_FP
from torch.utils.data import Dataset, DataLoader, sampler
from torchsummary import summary
import sys

class ContrastiveExtractor():

    def __init__(self, args, batch_size=1000, return_preactivation = True):


        self.batch_size = batch_size
        self.base_path = args.parentpath
        self.outfolder = args.outfolder

        self.get_wsi_path()
        
        self.wsi = openslide.OpenSlide(self.wsi_path)

        dataset = Whole_Slide_Bag_FP(file_path=self.patch_h5, wsi=self.wsi, pretrained=False, target_patch_size=224)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        kwargs = {'num_workers': 8, 'pin_memory': True} if self.device.type == "cuda" else {}
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=self.collate_features)

        self.model_path = args.modelpath
        self.return_preactivation = return_preactivation  # return features from the model, if false return classification logits
        # self.num_classes = 10  # only used if self.return_preactivation = False

        self.model = self.load_model()
        self.model = self.model.to(self.device)

        print("Initialized")

    def collate_features(self, batch):
        img = torch.cat([torch.from_numpy(item[0]) for item in batch], dim = 0)
        coords = np.vstack([item[1] for item in batch])
        return [img, coords]

    def get_wsi_path(self):

        parent_folder = Path(self.base_path)
        self.wsi_name = os.listdir(os.path.join(parent_folder, "data"))[0].replace(".svs", "")
        self.wsi_path = os.path.join(os.path.join(parent_folder, "data"), self.wsi_name + ".svs")
        self.patch_h5 = ""

        print("Using WSI File: ", self.wsi_name)
        print("With Abs Path: ", self.wsi_path)
        
        for root, dirs, files in os.walk(parent_folder):
            for f in files:
                if "blockmap.h5" in f:
                    self.patch_h5 = os.path.join(root, f)
                    print("Blockmap found: ", self.patch_h5)

                elif self.wsi_name + ".h5" in f:
                    self.patch_h5 = os.path.join(root, f)
                    print("Patch File found: ", self.patch_h5)

        if len(self.patch_h5) == 0:
            print("No CLAM h5 coords detected for file: ", self.wsi_path)
            print("Aborting...")
            sys.exit()
        

    def load_model(self):
        model = torchvision.models.__dict__['resnet18'](pretrained=False)

        state = torch.load(self.model_path, map_location='cuda:0')

        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model = self.load_model_weights(model, state_dict)

        if self.return_preactivation:
            model.fc = torch.nn.Sequential()
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)

        return model


    def load_model_weights(self, model, weights):

        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model

    def load_imgs(self, img_paths):
        try:
            return np.array([np.reshape(np.array(Image.open(img).convert('RGB').resize((224,224))), (3,224,224)) for img in img_paths])
        except:
            return np.array([])

    def extract_features(self, imgs):
        # image = np.array(Image.open(os.path.join(path, img_paths[0])))

        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor() 
        # Convert the image to PyTorch tensor 
        # tensor = transform(images)

        # print("Device:", device)
        tensor = torch.from_numpy(imgs).float().to(self.device)

        out = self.model(tensor)
        frame = pd.DataFrame(out.cpu().detach().numpy())

        return frame



    def get_wsi_paths(self):

        [print(x) for x in os.listdir(self.base_path)]
        wsi_paths = [os.path.join(self.base_path, x) for x in os.listdir(self.base_path)]
        print(wsi_paths)
        print("Loaded {0} WSI-Folder(s)".format(len(wsi_paths)))

        return wsi_paths



    def extract_features_from_patchfiles(self, wsi_path):

        dataframe = pd.DataFrame()

        data_path = os.path.join(wsi_path, "data")

        img_paths = [x for x in os.listdir(data_path)]
        self.img_paths = [os.path.join(data_path, x) for x in img_paths]
        
        for subset in tqdm(more_itertools.chunked(self.img_paths, self.batch_size)):
            
            imgs = self.load_imgs(subset)
            if len(imgs) > 0:
                frame = self.extract_features(imgs)
            else:
                frame = pd.DataFrame([])

            dataframe = pd.concat([dataframe, frame])

        print("OUT:")
        print(dataframe)

        dataframe.to_csv(os.path.join(wsi_path, "features_frame.csv"))


    def extract_features_from_h5file(self):

        with h5py.File(self.patch_h5, "r") as f:
            all_coords = np.array(f["coords"])

        all_feat_frame = pd.DataFrame([])
        chunked_list = list(more_itertools.chunked(all_coords, self.batch_size))
        print_every = 20

        for count, (batch, coords) in enumerate(self.loader):
            with torch.no_grad():	
                if count % print_every == 0:
                    print('batch {}/{}, {} files processed'.format(count, len(self.loader), count * self.batch_size))
                batch = batch.to(self.device, non_blocking=True)

                print("batch: ", batch.shape)
                
                features = self.model(batch)
                feat_frame = pd.DataFrame(features.cpu().numpy())
                coords_frame = pd.DataFrame(data=coords, columns=["x", "y"])

                feat_frame = pd.concat([coords_frame, feat_frame], axis=1, ignore_index=True)

                all_feat_frame = pd.concat([all_feat_frame, feat_frame], ignore_index=True)

        # for coord_subset in tqdm(chunked_list):

        #     patch_array = self.create_patch_dict(coord_subset)
        #     frame = self.extract_features(patch_array)

        csv_name = "{0}_features_frame.csv".format(self.wsi_name)

        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)

        all_feat_frame.to_csv(os.path.join(self.outfolder, csv_name))


    def get_patch(self, coords, wsi_path, patch_size=256):

        patch = self.wsi.read_region(tuple(coords), level=0, size=(patch_size, patch_size)).convert('RGB')
        patch = np.asarray(patch.resize((224,224))).transpose()

        # plt.figure()
        # plt.imshow(patch)

        # plt.figure()
        # plt.imshow(patch_)

        # plt.show()

        return patch

    def create_patch_dict(self, coords):
        patch_dict = {}
        count = 0
        # images = np.array([np.reshape(np.array(Image.open(img).convert('RGB').resize((224,224))), (3,224,224)) for img in img_paths])

        patch_array = np.zeros((self.batch_size, 3, 224, 224))
        for coord in coords:
            patch = self.get_patch(coord, self.wsi_path)
            patch_array[count] = patch
            count += 1

        return patch_array


if __name__ == "__main__":
    # images = torch.rand((10, 3, 224, 224), device='cuda')
    # self.model_path_ = '/home/user/Documents/Master/contrastive_learning/tenpercent_resnet18.ckpt'

    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', '--parentpath', type=str, required=False, default='/home/simon/philipp/HRD-Subset-I/DigitalSlide_A1M_9S_1_20190127165819218')
    parser.add_argument('-hp', '--patch_h5', type=str, required=False, default='')
    parser.add_argument('-o', '--outfolder', type=str, required=False, default='')
    parser.add_argument('-m', '--modelpath', type=str, required=False, default='/home/simon/philipp/checkpoints/tenpercent_resnet18.ckpt')

    args = parser.parse_args()

    base_path = args.parentpath

    # if len(base_path) > 0:
    #     ce = ContrastiveExtractor(args)
    #     for wsi_path in ce.wsi_paths:
    #         print("File: ", wsi_path)
    #         feat_file = os.path.join(wsi_path, "features_frame.csv")
            
    #         if os.path.isfile(feat_file):
    #             print("Features found")
    #             continue
    #         else:
    #             print("Calculating features...")
    #             ce.extract_features_from_patchfiles(wsi_path)

            
    # else:
    ce = ContrastiveExtractor(args)
    # print("Model:", summary(ce.model, (3,224,224)))
    ce.extract_features_from_h5file()



