import torchvision
import torch
import PIL
from PIL import Image 
import torchvision.transforms as transforms
import os
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

class ContrastiveExtractor():

    def __init__(self, base_path, batch_size=250, model_path="/home/simon/philipp/checkpoints/tenpercent_resnet18.ckpt", return_preactivation = True):


        self.batch_size = batch_size
        self.base_path = base_path

        print(self.base_path)

        if os.path.isfile(base_path):
            self.h5path = base_path
            self.get_wsi_path()
            self.wsi = openslide.OpenSlide(self.wsi_path)
            self.dataset = Whole_Slide_Bag_FP(file_path=h5path, wsi=self.wsi, pretrained=False, target_patch_size=224)
        else:   
            self.wsi_paths = self.get_wsi_paths()
            print(self.wsi_paths)

        self.model_path_ = model_path
        # self.model_path_ = '/home/user/Documents/Master/contrastive_learning/tenpercent_resnet18.ckpt'
        self.return_preactivation = return_preactivation  # return features from the model, if false return classification logits
        # self.num_classes = 10  # only used if self.return_preactivation = False

        self.model = self.load_model()

        print("Initialized")


    def get_wsi_path(self):

        parent_folder = self.h5path.split("/results")[0]
        self.wsi_name = parent_folder.split("/")[-1]
        self.wsi_path = os.path.join(os.path.join(parent_folder, "data"), self.wsi_name + ".svs")

        print("Using WSI File: ", self.wsi_name)
        print("With Abs Path: ", self.wsi_path)

        
    def load_model(self):
        model = torchvision.models.__dict__['resnet18'](pretrained=False)

        try:
            state = torch.load(self.model_path, map_location='cuda:0')
            # img_path = "/home/simon/philipp/patches/DigitalSlide_A1M_9S_1_20190127165819218"
        except:
            state = torch.load(self.model_path_, map_location='cuda:0')

        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model = self.load_model_weights(model, state_dict)

        if self.return_preactivation:
            model.fc = torch.nn.Sequential()
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)

        return model.cuda()


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

        except PIL.UnidentifiedImageError as e:

            print("PIL Error: ", e)
            print("Skipping batch...")
        
            return np.array([])

    def extract_features(self, imgs):

        # image = np.array(Image.open(os.path.join(path, img_paths[0])))

            # Define a transform to convert the image to tensor
            transform = transforms.ToTensor() 
            # Convert the image to PyTorch tensor 
            # tensor = transform(images)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print("Device:", device)
            tensor = torch.from_numpy(imgs).float().to(device)

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

        with h5py.File(self.h5path, "r") as f:
            all_coords = np.array(f["coords"])

        all_feat_frame = pd.DataFrame([])
        chunked_list = list(more_itertools.chunked(all_coords, self.batch_size))

        loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)

        for count, (batch, coords) in tqdm(enumerate(loader)):
            batch = batch.to(device, non_blocking=True)
            features = self.model(batch)
            # patch_array = self.create_patch_dict(coord_subset)
            # frame = self.extract_features(patch_array)
            features = pd.DataFrame(features.cpu().numpy())
            all_feat_frame = pd.concat([all_feat_frame, frame])

        all_feat_frame.to_csv(os.path.join("features_frame.csv"))

    def get_patch(self, coords, wsi_path, patch_size=256):

        patch = self.wsi.read_region(tuple(coords), level=0, size=(patch_size, patch_size)).convert('RGB')
        patch = np.reshape(patch.resize((224,224)), (3,224,224))


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

    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', '--parentpath', type=str, required=False, default='')
    parser.add_argument('-hp', '--h5path', type=str, required=False, default='/media/user/easystore/HRD-Subset-I/DigitalSlide_A1M_1S_1_20190127153208246/results/7fdd6b3355754fe88db4ebfc72f2c0b8/raw/heat/Unspecified/DigitalSlide_A1M_1S_1_20190127153208246/DigitalSlide_A1M_1S_1_20190127153208246_blockmap.h5')
    args = parser.parse_args()

    base_path = args.parentpath
    h5path = args.h5path

    if len(base_path) > 0:
        ce = ContrastiveExtractor(base_path)
        for wsi_path in ce.wsi_paths:
            print("File: ", wsi_path)
            feat_file = os.path.join(wsi_path, "features_frame.csv")
            
            if os.path.isfile(feat_file):
                print("Features found")
                continue
            else:
                print("Calculating features...")
                ce.extract_features_from_patchfiles(wsi_path)

            
    elif len(h5path) > 0:
        ce = ContrastiveExtractor(h5path)
        ce.extract_features_from_h5file()



