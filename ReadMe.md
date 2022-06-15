# Readme
This repo is based on functions from CLAM and the "SimCLR" - contrastive learning paper(s):

CLAM: https://github.com/mahmoodlab/CLAM
SimCLR (for HistoPathology): https://github.com/ozanciga/self-supervised-histopathology

This file describes the basic idea of every script in this repo.

## analyse_feats.py
applies kmeans algorithm to specified data and saves the trained model



## contrastive.py
Extracts features from WSI using the SimCLR pretrained ResNet50 model



## convert_png.py
converts all images stored in one directory to png

## create_dir.py

moves WSI files of type SVS from parent folder into paent/data
E.G.: /parent/wsi-x.svs   -->    /parent/wsi-x/data/wsi-x.svs
this is done so the docker containers can be used more easily (mounting the whole data folder)

## main.py

reads a given HDF5 file with features (extracted using the contrastive.py script or the corresponding docker container)
reads the corresponding target file
training can be done with the parsed data (multilayer perceptron, TSNE visualization, SVM and other functions implemented)

## patch_post_process.py

extracts patch image files from a WSI. The extracted patches can be thresholded by their corresponding attention scores
calculated by CLAM (e.g.: export all patches with attention score > threshold)

creates a PDF with exported patches and classified patchey by HoVerNet
but this function is BROKEN (see "process_files" function for more details)

## post_proc_heatmap.py


## process_output.py

reads a number of whole slide images and extracts a certain area ("Super Patch"). The area is chosen according to the attention score values
of all single patches within this area. The largest areas that exceed a certain attention score can be extracted and saved into files.


## tf_models.py
implemented tensorflow models:  multilayer perceptrons for classification and regression

## wsi_loader.py
specific WSI loader from CLAM for tensorflow training routines