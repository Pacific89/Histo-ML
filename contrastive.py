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


class ContrastiveExtractor():

    def __init__(self, base_path, batch_size=1000):


        self.batch_size = batch_size
        self.base_path = base_path

        self.wsi_paths = self.get_wsi_paths()
        self.model_path = "/home/simon/philipp/checkpoints/tenpercent_resnet18.ckpt"
        # self.model_path_ = '/home/user/Documents/Master/contrastive_learning/tenpercent_resnet18.ckpt'
        self.return_preactivation = True  # return features from the model, if false return classification logits
        # self.num_classes = 10  # only used if self.return_preactivation = False

        self.model = self.load_model()

        print(self.base_path)
        print(self.wsi_paths)
        print("Initialized")



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


    def load_extract(self, img_paths):

        # image = np.array(Image.open(os.path.join(path, img_paths[0])))
        try:
            images = np.array([np.reshape(np.array(Image.open(img).convert('RGB').resize((224,224))), (3,224,224)) for img in img_paths])

            # Define a transform to convert the image to tensor
            transform = transforms.ToTensor() 
            # Convert the image to PyTorch tensor 
            # tensor = transform(images)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print("Device:", device)
            tensor = torch.from_numpy(images).float().to(device)

            out = self.model(tensor)
            frame = pd.DataFrame(out.cpu().detach().numpy(), index=img_paths)

            del out
            del images
            del tensor
            del img_paths

            return frame

        except PIL.UnidentifiedImageError as e:

            print("PIL Error: ", e)
            print("Skipping batch...")
        
            return pd.DataFrame([])



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
            
            frame = self.load_extract(subset)

            dataframe = pd.concat([dataframe, frame])

        print("OUT:")
        print(dataframe)

        dataframe.to_csv(os.path.join(wsi_path, "features_frame.csv"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=False)
    args = parser.parse_args()

    base_path = args.path
    ce = ContrastiveExtractor(base_path)
    # images = torch.rand((10, 3, 224, 224), device='cuda')


    for wsi_path in ce.wsi_paths:
        print("File: ", wsi_path)
        feat_file = os.path.join(wsi_path, "features_frame.csv")
        
        if os.path.isfile(feat_file):
            print("Features found")
            continue
        else:
            print("Calculating features...")
            ce.extract_features_from_patchfiles(wsi_path)



