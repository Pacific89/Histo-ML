import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import pickle
import argparse

class FeatureAnalysis():
    """ class for training and analysing kmeans algorithms
    """

    def __init__(self, path, keep, server=False):
        """ init method to set some parameters

        Parameters
        ----------
        path : string
            parent path for kmeans models to be store or loaded from
        keep : int
            number of files to keep for kmeans training
        server : bool, optional
            info if started on server (for training), by default False
            if run on server, loads the list of feature frames and stores them into one large dataframe
        """

        self.parent_path = path
        self.random_state = 0

        if server:
            self.frame_list = self.get_paths(keep_files=keep)
            self.all_feat_frame = self.create_dataframe()

    def run(self, k):
        """ function used to train multiple kmeans models consecutively
            can be called using a loop, iterating over several "k"s and one initialized
            FeatureAnalysis Object

            sets amount of clusters (k)
            trains kmeans model (calc_kmeans)
            plots kmeans results (plot_kmeans)
            finally saves the trained model (save_model)

        Parameters
        ----------
        k : int
            number of clusters for kmeans training
        """

        self.k = k
        self.kmeans = self.calc_kmeans()
        self.plot_kmeans()
        self.save_model(self.kmeans)

    def save_model(self, model):
        """ saves trained kmeans model as .pkl file

        Parameters
        ----------
        model : Object
            sklearn kmeans object
        """

        model_name = "kmeans_{0}.pkl".format(self.k)

        with open(model_name, "wb") as f:
            pickle.dump(model, f)


    def get_paths(self, keep_files=0):
        """ generates a list of paths to read the features_frame.csv from
            TODO the file type has changed to HDF5 format.
            This needs to be adapted (no more csv viles available!)

        Parameters
        ----------
        keep_files : int, optional
            files to keep for kmeans training. If 0, all files are used, by default 0

        Returns
        -------
        list
            list of paths to existing "feature_frame.csv" files
        """

        frame_list = []
        for folder in os.listdir(self.parent_path):
            feat_frame = os.path.join(self.parent_path, os.path.join(folder, "features_frame.csv"))
            print(feat_frame)
            if os.path.isfile(feat_frame):
                frame_list.append(feat_frame)

        if keep_files > 0:
            print("Reducing feature files from: {0} to: {1}".format(len(frame_list), keep_files))
            frame_list = frame_list[:keep_files]

        return frame_list


    def create_dataframe(self):
        """ creates empty dataframe and appends all feature frames

        Returns
        -------
        Pandas DataFrame
            dataframe with all features for all used files (from features_frame.csv files)
        """

        all_feat_frame = pd.DataFrame([])
        print("Getting features...")
        for frame_file in tqdm(self.frame_list):
            frame = pd.read_csv(frame_file)
            all_feat_frame = pd.concat([all_feat_frame, frame])

        print("Features Shape: (patches, fileName + features)", all_feat_frame.values.shape)

        return all_feat_frame

    def check_kmeans(self):
        """ loads multiple kmeans models stored in self.parent_path and plots cumulative distances (y-axis)
            for each kmeans models with different amounts of clusters (x-axis)
            kmeans models are expected to be stored as .pkl files
        """
        print("Checking kmeans models:")

        kmeans_paths = [os.path.join(self.parent_path, f) for f in os.listdir(self.parent_path) if f.endswith('.pkl')]
        # print(kmeans_paths)
        res = list()
        n_cluster = list()
        for kmeans_path in kmeans_paths:
            with open(kmeans_path, 'rb') as km:
                model = pickle.load(km)
            
            res.append(model.inertia_)
            n_cluster.append(len(model.cluster_centers_))


        print("Samples: ", len(model.labels_))
        print(res)
        print(n_cluster)
        ind = np.argsort(np.array(n_cluster).astype(int))
        print(type(ind))
        plt.rcParams.update({'font.size': 22})
        plt.plot(np.array(n_cluster)[ind], np.array(res)[ind], label="Distances")
        plt.title('KMeans Clustering')
        # plt.savefig("elbow_curve.png")
        plt.ylabel('Cumulative Distances')
        plt.xlabel('Number of Clusters')
        # plt.vlines(200, min(res), max(res), colors=["red"], linestyles=["dashed"], label="Elbow reported by ")
        plt.show()

    def calc_kmeans(self):
        """ prepares data from the feature frame and computes a kmeans model with the set number of clusters

        Returns
        -------
        Object
            trained kmeans sklearn object 
        """
        self.data = self.all_feat_frame.iloc[:, 1:].values
        self.paths = [d[0] for d in self.all_feat_frame.iloc[:, :1].values]

        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state).fit(self.data)

        return kmeans

    def plot_kmeans(self):
        """ plots the image of the cluster center and the 8 Images
            with features closest to the cluster center.
            Finally saves the figure as png
        """

        centroids = self.kmeans.cluster_centers_
        # print(centroids)
        dirname = "{0}_clusters".format(self.k)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        for n, center in enumerate(centroids):
            fig, ax = plt.subplots(3,3)
            
            distances = np.argsort(np.linalg.norm(self.data - center, axis=1))
            cen = distances[0]
            fig.suptitle("Center: {0} | Patch: {1}".format(n, self.paths[cen]))

            for ind, a in enumerate(ax.flatten()):
                dist = distances[ind]
                # print(paths[dist])
                with Image.open(self.paths[dist]) as im:
                    patch_img = im.copy()


                a.imshow(patch_img)
                a.axis("off")
            # ax[int(n%3), n].imshow(patch_img)

            plt.savefig(os.path.join(dirname, "cluster_{0}.png".format(n)))

            # plt.show()

        

if __name__ == "__main__":
        """ main function provides some command line arguments and starts the kmeans functions accordingly.
            k_clusters: number of clusters for kmeans algorithm
            keep: number of files to use for kmeans training
            load_kmeans: load existing kmeans object
            server: True if started on server (used for training, assuming that server has much more RAM)

            creates an object from class "FeatureAnalysis" which provides the function for kmeans training and analysis
        """

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_clusters', required=False, nargs='+', default=[200])
    parser.add_argument('-ke', '--keep', type=int, required=False, default=0)
    parser.add_argument('-lk', '--load_kmeans', required=False, type=str, default='')
    parser.add_argument('-s', '--server', type=bool, required=False, default=False)
    args = parser.parse_args()

    k_clusters = args.k_clusters
    keep = args.keep
    kmeans_path = args.load_kmeans

    if args.server:
        path = "/home/simon/philipp/patches"
        fa = FeatureAnalysis(path, keep=keep, server=True)
        for k in k_clusters:
            print("kmeans with: ", int(k))
            fa.run(int(k))

    else:
        if len(kmeans_path) > 0:
            path = kmeans_path
            fa = FeatureAnalysis(path, keep=keep)
            fa.check_kmeans()


    # patch_path = "/media/user/easystore/patches/"
    # patch_path_ = "/media/user/easystore/DigitalSlide_A1M_9S_1_20190127165819218/"
    # csv_path = "/home/user/Documents/Master/docker/features_frame.csv"
    # csv_path_low = "/home/user/Documents/Master/docker/features_frame_low_att.csv"


    # num_patches = -1

    # data_ = pd.read_csv(csv_path)
    # data = data_.iloc[:, 1:].values[:num_patches]

    # data_low_ = pd.read_csv(csv_path_low)
    # data_low = data_low_.iloc[:, 1:].values[:num_patches]

    # paths = [p[0].replace("/home/simon/philipp/", patch_path) for p in data_.iloc[:, :1].values]
    # paths_low = [p[0].replace("/home/simon/philipp/patches/", patch_path) for p in data_low_.iloc[:, :1].values]


    # all_data = np.concatenate((data[:num_patches], data_low[:num_patches]))
    # all_paths = np.concatenate((paths[:num_patches], paths_low[:num_patches]))


    # # k = check_kmeans(all_data, all_paths)
    # kmeans_plot(all_data, all_paths)
    