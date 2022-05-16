from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import os
import h5py
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import umap
import umap.plot
import pickle

from sklearn.model_selection import train_test_split


def umap_func(X_train, X_test, y_train, y_test):

    kmeans = load_kmeans()
    X_train = X_train.values.astype(float)
    y_pred = kmeans.predict(X_train)
    samples_use = 5
    num_clusters = 10
    num_features = X_train.shape[1]

    x_train_clusters = [] # np.zeros((int(samples_use*num_clusters) , num_features))
    y_train_clusters = [] # np.zeros((int(samples_use*num_clusters), 1))
    for num, centroid in enumerate(kmeans.cluster_centers_[:num_clusters]):
        label = kmeans.labels_[num]
        distances = np.abs(np.sum((centroid - X_train), axis=1))
        # print(distances.shape)
        indices = np.argsort(distances)
        # print(indices)
        ind_use = indices[:samples_use]

        # x_train_clusters[num*samples_use : (num+1)*samples_use] = X_train[ind_use]
        # y_train_clusters[num*samples_use : (num+1)*samples_use] = np.array([label] * samples_use).reshape(samples_use,1)
        x_train_clusters.append(X_train[ind_use])
        y_train_clusters.append(kmeans.labels_[ind_use])
        print(label)

    mapper = umap.UMAP().fit(np.array(x_train_clusters).reshape(int(num_clusters*samples_use), num_features))

    umap.plot.points(mapper, labels=np.array(y_train_clusters).flatten(), theme='fire')
    umap.plot.plt.imsave("umap.png")


def sgd_reg_func(X_train, X_test, y_train, y_test):
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    reg.fit(X_train, y_train)

    print(reg.score(X_test, y_test))

def mlp_regressor(X_train, X_test, y_train, y_test):
    from tf_models import _mlp_regressor
    _mlp_regressor(X_train, y_train)

def mlp_classifier(X_train, X_test, y_train, y_test):
    from tf_models import _mlp_classifier
    _mlp_classifier(X_train, y_train)

def svm_func(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)

def tsne_func(X_train, X_test, y_train, y_test):
    palette = sns.color_palette("bright", len(set(y_train)))
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X_train)

    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_train, legend='full', palette=palette)
    plt.savefig("tsne.pdf")

def load_kmeans():
    model = pickle.load(open("/media/user/easystore/ContrastiveClusterResults/kmeans_tests/kmeans_200.pkl", "rb"))
    return model



def get_combined_data(args):

    combined_features = pd.DataFrame()
    combined_targets = pd.DataFrame()

    with pd.ExcelFile(args.xlsx_path) as xlsx:
        # xlsx = pd.ExcelFile(args.csv)
        worksheet = pd.read_excel(xlsx, "codiert")

        for c, row in worksheet.iterrows():
            simclr_paths = row["simclr_results"]
            if pd.notna(simclr_paths):
                simclr_paths = simclr_paths.replace("[", "").replace("'", "").replace("]", "")

                for path in simclr_paths.split(","):
                    name = path.split("/results")[0].split("/")[-1]
                    # print(name)
                    filepath = os.path.join(path.replace(" ", ""), name + "_features_frame.h5")
                    if os.path.isfile(filepath):
                        print(filepath)
                        with h5py.File(filepath, "r") as f:
                            features_ = pd.DataFrame(np.array(f["features"]))

                            targets_ = [row["HRD-Status"]]*len(features_)
                            # targets_ = [row["GIS-Wert"]]*len(features_)

                            use_samples = int(len(features_)/1)
                            features = features_[:use_samples]
                            targets = pd.DataFrame(targets_[:use_samples])

                            combined_features = pd.concat([combined_features, features], ignore_index=True)
                            combined_targets = pd.concat([combined_targets, targets], ignore_index=True)


    combined_targets = combined_targets.values.flatten()

    return combined_features, combined_targets

def get_combined_data_subset(args):

    combined_features = pd.DataFrame()
    combined_targets = pd.DataFrame()

    with pd.ExcelFile(args.xlsx_path) as xlsx:
        # xlsx = pd.ExcelFile(args.csv)
        worksheet = pd.read_excel(xlsx, "Sheet1")
        # print(worksheet)
        for c, row in worksheet.iterrows():
            filename = row["filename"]
            if pd.notna(filename):
                print("File: ", filename)
                data_path = os.path.join(args.parent_path, filename.split(".svs")[0])
                if os.path.isdir(data_path):
                    for r, d, f in os.walk(data_path):
                        for file_ in f:
                            if file_.endswith("features_frame.h5"):
                                filepath = os.path.join(r, file_)
                                print("Found:")
                                print(filepath)
                                with h5py.File(filepath, "r") as f:
                                    features_ = pd.DataFrame(np.array(f["features"]))

                                    targets_ = [row["HRD-Status"]]*len(features_)
                                    # targets_ = [row["GIS-Wert"]]*len(features_)

                                    use_samples = int(len(features_)/1)
                                    features = features_[:use_samples]
                                    targets = pd.DataFrame(targets_[:use_samples])

                                    combined_features = pd.concat([combined_features, features], ignore_index=True)
                                    combined_targets = pd.concat([combined_targets, targets], ignore_index=True)


    combined_targets = combined_targets.values.flatten()

    return combined_features, combined_targets

if __name__ == "__main__":
    """main function that handles input arguments, reads h5 feature frame files (from the results paths of the xlsx file)
    
    1) Run feature extraction (CLAM patches & SIMCLR features) using "start_wrapper.py" of https://github.com/Pacific89/docker
    2) The "examples_hrd.xlsx" file contains paths to CLAM and SIMCLR results folders
    3) Call this main.py script with the new (changed) "examples_hrd.xlsx" file ("python main.py -p examples_hrd.xlsx")
    4) This script constructs one dataframe from the input feature frames
    5) Finally the different functions for ML and data analysis can be used
    """

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xlsx_path', required=False, default="/media/user/easystore/HRDDATA/hrd_subset.xlsx")
    parser.add_argument('-p', '--parent_path', required=False, default="/media/user/easystore/HRDDATA")
    args = parser.parse_args()


    # get all features and corresponding targets
    combined_features, combined_targets = get_combined_data_subset(args)
                            
    # print(combined_features)
    print(combined_targets)


    # split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_targets, test_size=0.2, random_state=42)



    # Call the different ML / data analysis functions

    tsne_func(X_train, X_test, y_train, y_test)
    # svm_func(X_train, X_test, y_train, y_test)
    # mlp_classifier(X_train, X_test, y_train, y_test)
    # mlp_regressor(X_train, X_test, y_train, y_test)
    # umap_func(X_train, X_test, y_train, y_test)
    # sgd_reg_func(X_train, X_test, y_train, y_test)


