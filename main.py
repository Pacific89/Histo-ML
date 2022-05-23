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
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
import umap
import umap.plot
import pickle
import shutil
import shortuuid

from sklearn.model_selection import train_test_split


class ML():
    """Class for providing several machine learning functions that can be used
    in a "standartized" way: the object is initialized with features and all targets (for classification and regression)
    and the specific function load the values they need from the class attributes.

    E.G.: X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class)
    """
    
    def __init__(self, args, combined_features, combined_targets_class, combined_targets_reg):

        self.combined_features = combined_features
        self.combined_targets_class = combined_targets_class
        self.combined_targets_reg = combined_targets_reg
        self.exp_base_path = os.path.abspath(args.exp_base_path)

    def umap_func(self):
        """UMAP function from package: https://umap-learn.readthedocs.io/en/latest/basic_usage.html
        generates a dimensionality reduced output figure that show the projection of high dimensional features
        """
        exp_folder = os.path.join(self.exp_base_path, "exp_" + shortuuid.uuid()[:8])
        os.makedirs(exp_folder)
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        mapper = umap.UMAP().fit(X_train)

        umap.plot.points(mapper, labels=y_train, theme='fire')
        umap.plot.plt.imsave(os.path.join(exp_folder, "umap.png"))


    def sgd_reg_func(self):
        """sklearn SGD-regressor: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
        regression model to learn continous values
        """
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_reg, test_size=0.2, random_state=42)

        reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
        reg.fit(X_train, y_train)

        print(reg.score(X_test, y_test))

    def mlp_regressor(self):
        """Multilayer Perceptron implementation of Tensorflow.
        See tf_models.py for more infos
        """
        from tf_models import _mlp_regressor

        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_reg, test_size=0.2, random_state=42)

        _mlp_regressor(X_train, y_train)

    def mlp_classifier(self):
        """Multilayer Perceptron implementation of Tensorflow.
        See tf_models.py for more infos
        """
        from tf_models import _mlp_classifier

        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        _mlp_classifier(X_train, y_train)

    def check_models(self):
        """ generate different multi layer perceptron models using the function "get_models" from "tf_models.py"
        to find best architectures.
        See tf_models.py for more infos

        """
        from tf_models import get_models, _mlp_classifier
        models = get_models(4, 32, 192, 32, 512)
        print("Checking {0} Models for Classification...".format(len(models)))
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        for model in models:
            _mlp_classifier(X_train, y_train, epochs=10, model=model)

    def svm_func(self):
        """Support Vector Machine for classification tasks (not enough memory with n_jobs=20 and n_estimators=20 using "medum dataset")
        """
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        # clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
        clf = BaggingClassifier(base_estimator=svm.SVC(gamma='auto'), n_jobs=20, n_estimators=20, random_state=42).fit(X_train, y_train)
        # clf.fit(X_train, y_train)

        scores = clf.score(X=X_test, y=y_test)
        print("SVM Score: ", scores)

    def tsne_func(self):
        """Dimensionality reduction using tsne from sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        outputs a PDF plotting the samples on a reduced feature space
        """
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        palette = sns.color_palette("bright", len(set(y_train)))
        tsne = TSNE()
        X_embedded = tsne.fit_transform(X_train)

        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_train, legend='full', palette=palette)
        plt.savefig("tsne.pdf")

    def naive_bayes_estimator(self):
        """Bayes Estimator from: https://scikit-learn.org/stable/modules/naive_bayes.html
        for efficient classification (using a lot of samples)
        Less computionally expensive than SVM above.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.combined_features, self.combined_targets_class, test_size=0.2, random_state=42)

        gnb = GaussianNB()
        gnb_classifier = gnb.fit(X_train, y_train)
        score = gnb_classifier.score(X_test, y_test)
        print("Score: ", score)


    def load_kmeans(self):
        model = pickle.load(open("/media/user/easystore/ContrastiveClusterResults/kmeans_tests/kmeans_200.pkl", "rb"))
        return model



def save_h5_files(args, combined_features, combined_targets_class, combined_targets_reg):
    """save the combined features and targets to three distinct HDF5 files
    Parameters
    ----------
    args : dictionary
        command line arguments
    combined_features : data frame
        data frame storing features
    combined_targets_class : data frame
        data frame storing classification targets
    combined_targets_reg : data frame
        data frame storing regression targets
    """

    dataset_path = "data/dataset_{0}".format(len(os.listdir("data")))
    os.makedirs(dataset_path)

    shutil.copy2(args.xlsx_path, dataset_path)
    combined_features.to_hdf(os.path.join(dataset_path, "combined_features.h5"), "simclr_features")
    combined_targets_class.to_hdf(os.path.join(dataset_path, "combined_hrd_targets.h5"), "hrd_class_targets")
    combined_targets_reg.to_hdf(os.path.join(dataset_path, "combined_gis_targets.h5"), "gis_score_targets")


def get_combined_data_subset(args):
    """ reads the single feature frame HDF5 files, combines them into one HDF5 file and stores it.

    Parameters
    ----------
    args : dict
        command line arguments dictionary

    Returns
    -------

    combined_features, combined_targets_class, combined_targets_reg: data frames
        data frames that hold the features and targets respectively
    """

    combined_features = pd.DataFrame()
    combined_targets_class = pd.DataFrame()
    combined_targets_reg = pd.DataFrame()

    # Open excel file with file paths
    with pd.ExcelFile(args.xlsx_path) as xlsx:
        # xlsx = pd.ExcelFile(args.csv)
        worksheet = pd.read_excel(xlsx, "Sheet1")
        
        # iterate over each row (file path)
        for c, row in worksheet.iterrows():
            filename = row["filename"]
            if pd.notna(filename):
                print("File: ", filename)
                data_path = os.path.join(args.parent_path, filename.split(".svs")[0])
                if os.path.isdir(data_path):
                    # find the correct features files for each WSI file path:
                    for r, d, f in os.walk(data_path):
                        for file_ in f:
                            if file_.endswith("features_frame.h5"):
                                filepath = os.path.join(r, file_)
                                print("Found:")
                                print(filepath)

                                # Open the HDF5 features files and store features/targets to pandas data sets
                                with h5py.File(filepath, "r") as f:
                                    features_ = pd.DataFrame(np.array(f["features"]))

                                    targets_class = [row["HRD-Status"]]*len(features_)
                                    targets_reg = [row["GIS-Wert"]]*len(features_)

                                    use_samples = int(len(features_)/1)
                                    features = features_[:use_samples]
                                    targets_class = pd.DataFrame(targets_class[:use_samples])
                                    targets_reg = pd.DataFrame(targets_reg[:use_samples])

                                    combined_features = pd.concat([combined_features, features], ignore_index=True)
                                    combined_targets_class = pd.concat([combined_targets_class, targets_class], ignore_index=True)
                                    combined_targets_reg = pd.concat([combined_targets_reg, targets_reg], ignore_index=True)


    save_h5_files(args, combined_features, combined_targets_class, combined_targets_reg)

    combined_targets_class = combined_targets_class.values.flatten()
    combined_targets_reg = combined_targets_reg.values.flatten()

    return combined_features, combined_targets_class, combined_targets_reg


def check_datasets(args):
    """check if the data set (single files) have already been stored as HDF5 file

    TODO check independent of order!!!! (so far only the file lists are compared)

    Parameters
    ----------
    args : dict
        command line arguments dictionary

    Returns
    -------
    dataset_found: bool
        True/False according to data set found: Yes/No
    dataset_path: string
        path to the HDF5 file
    """

    dataset_found = False
    dataset_path = ""

    input_filenames = pd.read_excel(args.xlsx_path)["filename"]
    for root, dirs, files in os.walk("data"):
        for f in files:
            if f.endswith("xlsx"):
                found_excel_filenames = pd.read_excel(os.path.join(root, f))
                print("Excel found!")
                if len(input_filenames) == len(found_excel_filenames["filename"]):
                    same_files = sum(input_filenames == found_excel_filenames["filename"])
                    print("Same files: ", same_files)

                    if same_files == len(input_filenames):
                        dataset_path = root
                        dataset_found = True

    return dataset_found, dataset_path

if __name__ == "__main__":
    """main function that handles input arguments, reads h5 feature frame files (from the results paths of the xlsx file)
    
    1) Run feature extraction (CLAM patches & SIMCLR features) using "start_wrapper.py" of https://github.com/Pacific89/docker
    2) The "examples_hrd.xlsx" file contains paths to CLAM and SIMCLR results folders
    3) Call this main.py script with the new (changed) "examples_hrd.xlsx" file ("python main.py -p examples_hrd.xlsx")
    4) This script constructs one dataframe from the input feature frames
        4a) If the data is new (not saved before) the resulting dataframe is save as HDF5 file
        4b) If the data has been used before, the corresponding HDF5 file is loaded
    5) Finally the different functions for ML and data analysis can be used

    TODO command line arguments could/should be replaced by config file
    """

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xlsx_path', required=False, default="")
    parser.add_argument('-p', '--parent_path', required=False, default="")
    parser.add_argument('-mr', '--mlp_reg', required=False, default=False)
    parser.add_argument('-mc', '--mlp_class', required=False, default=False)
    parser.add_argument('-svm', '--svm_class', required=False, default=False)
    parser.add_argument('-t', '--tsne_class', required=False, default=False)
    parser.add_argument('-uc', '--umap_class', required=False, default=False)
    parser.add_argument('-sr', '--sgd_reg', required=False, default=False)
    parser.add_argument('-nb', '--naive_bayes', required=False, default=False)
    parser.add_argument('-cm', '--check_models', required=False, default=False)
    parser.add_argument('-dp', '--data_path', required=False, default="")
    parser.add_argument('-ep', '--exp_base_path', required=False, default="results")

    args = parser.parse_args()
    data_path = ""
    if len(args.xlsx_path) > 0:
        dataset_found, data_path = check_datasets(args)

        if dataset_found:
            print("Found Dataset at: ", data_path)
            print("Files: ", os.listdir(data_path))

        else:
            combined_features, combined_targets_class, combined_targets_reg = get_combined_data_subset(args)

    if os.path.isdir(args.data_path):
        data_path = args.data_path


    if len(data_path) > 0 and os.path.isdir(data_path):
        features_path = "combined_features.h5"
        targets_class_path = "combined_hrd_targets.h5"
        targets_reg_path = "combined_gis_targets.h5"

        datapath = os.path.abspath(data_path)
        combined_features = pd.read_hdf(os.path.join(datapath, features_path), key="simclr_features")
        combined_targets_class = pd.read_hdf(os.path.join(datapath, targets_class_path), key="hrd_class_targets")
        combined_targets_reg = pd.read_hdf(os.path.join(datapath, targets_reg_path), key="gis_score_targets")

    # print(combined_targets_reg)
    print(combined_targets_class)
    print(combined_features)

    # Call the different ML / data analysis functions
    ml = ML(args, combined_features, combined_targets_class, combined_targets_reg)
    if args.tsne_class:
        ml.tsne_func()

    if args.svm_class:
        ml.svm_func()

    if args.mlp_class:
        ml.mlp_classifier()

    if args.mlp_reg:
        ml.mlp_regressor()

    if args.umap_class:
        ml.umap_func()

    if args.sgd_reg:
        ml.sgd_reg_func()

    if args.naive_bayes:
        ml.naive_bayes_estimator()
    
    if args.check_models:
        ml.check_models()


