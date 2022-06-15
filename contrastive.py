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
    """ class for using the SimCLR pretrained resnet from:
        https://github.com/ozanciga/self-supervised-histopathology

        Sets Input, Output and modelpath (from command line arguments)
        loads and prepares tensorflow model so the function "extract_fetaures_from_h5file"
        can be called.
    """

    def __init__(self, args, batch_size=1000, return_preactivation = True):
        """
        IMPORTAND - Required file structure: /path/to/WSI-x/data/wsi-x.svs
        Functions of the ContrastiveExtractor object can be used inside the SimCLR docker container:
        
        - BUILD docker image from docker file
        - RUN feature extraction: docker run --rm -v /path/to/WSI-x:/usr/local/mount simclr-docker

        OR: python3 /usr/local/src/contrastive.py -pp path/to/WSI-x -o output/path -m path/to/model

        Sets input,output and model paths
        opens WSI object and prepares a dataloader and the tensorflow model


        Parameters
        ----------
        args : dictionary
            command line arguments
        batch_size : int, optional
            batch size for feature extraction, by default 1000
        return_preactivation : bool, optional
            return values of the last layer if True else return class prediction (integer between 0 and num_classes), by default True
        """


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
        self.num_classes = 10  # arbitrary number!!! only used if self.return_preactivation = False

        self.model = self.load_model()
        self.model = self.model.to(self.device)

        print("Initialized")

    def collate_features(self, batch):
        """function used by the data loader
        
        Parameters
        ----------
        batch : batch object
            created by data loader

        Returns
        -------
        List
            image and coordinates for the Dataloader
        """
        img = torch.cat([torch.from_numpy(item[0]) for item in batch], dim = 0)
        coords = np.vstack([item[1] for item in batch])
        return [img, coords]

    def get_wsi_path(self):
        """ get paths for the WSI (clam coordinates stored in "blockmap.h5" or patch file "wsi-x.h5" and .svs file)
        if no patch file or blockmap is found, the script exits
        path to patch file is store in "self.patch_h5"
        """

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
        """ load tensorflow resnet18 model and load weights from the provided model path
            SimCLR model from: https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent

        Returns
        -------
        tensor flow resnet18 model with loaded SimCLR weights
        """
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
        """ loads weights

        Returns
        -------
        [type]
            [description]
        """

        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model



    def save_hdf5(self, output_path, asset_dict, attr_dict= None, mode='a'):
        """CLAMs hdf5 save function. stores values in hdf5 files using "update" mode by default

        Parameters
        ----------
        output_path : string
            path for output results
        asset_dict : dictionary
            asset_dict = {'features': features, 'coords': coords}
        attr_dict : dictionary, optional
            attributes can be specified, by default None
        mode : str, optional
            mode for writing the hdf5 file, by default 'a'
        """

        file = h5py.File(output_path, mode)
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
        file.close()

        return output_path


    def extract_features_from_h5file(self):
        """ use patch coordinates for the fetaure extraction
        and store the features in hdf5 files (calling save_hdf5())

        iterates over the batches generated by the data loader
        and applies the loaded model to each batch
        """

        with h5py.File(self.patch_h5, "r") as f:
            all_coords = np.array(f["coords"])

        all_feat_frame = pd.DataFrame([])
        chunked_list = list(more_itertools.chunked(all_coords, self.batch_size))
        print_every = 20

        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)

        h5_name = "{0}_features_frame.h5".format(self.wsi_name)
        output_path = os.path.join(self.outfolder, h5_name)
        
        mode = 'w'

        for count, (batch, coords) in enumerate(self.loader):
            with torch.no_grad():	
                if count % print_every == 0:
                    print('batch {}/{}, {} files processed'.format(count, len(self.loader), count * self.batch_size))
                batch = batch.to(self.device, non_blocking=True)

                print("batch: ", batch.shape)
                
                features = self.model(batch).cpu().numpy()

                # write to hdf5 from CLAM:
                asset_dict = {'features': features, 'coords': coords}
                self.save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'


if __name__ == "__main__":
        """
        IMPORTAND - Required file structure: /path/to/WSI-x/data/wsi-x.svs

        Functions of the ContrastiveExtractor object can be used inside the SimCLR docker container:
        
        - BUILD docker image from docker file
        - RUN feature extraction: docker run --rm -v /path/to/WSI-x:/usr/local/mount simclr-docker

        OR: python3 /usr/local/src/contrastive.py -pp path/to/WSI-x -o output/path -m path/to/model

        main function providing some command line arguments to specify input, output and modelpaths
        creates an Object of class "contrastiveExtractor" and uses its function "extract_features_from_h5file"
        images patches are read using patch coordinates from clam (from hdf5 files)
        """
    # images = torch.rand((10, 3, 224, 224), device='cuda')
    # self.model_path_ = '/home/user/Documents/Master/contrastive_learning/tenpercent_resnet18.ckpt'

    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', '--parentpath', type=str, required=False, default='/home/simon/philipp/HRD-Subset-I/DigitalSlide_A1M_9S_1_20190127165819218')
    parser.add_argument('-hp', '--patch_h5', type=str, required=False, default='')
    parser.add_argument('-o', '--outfolder', type=str, required=False, default='')
    parser.add_argument('-m', '--modelpath', type=str, required=False, default='/home/simon/philipp/checkpoints/tenpercent_resnet18.ckpt')

    args = parser.parse_args()

    ce = ContrastiveExtractor(args)
    # print("Model:", summary(ce.model, (3,224,224)))
    ce.extract_features_from_h5file()



