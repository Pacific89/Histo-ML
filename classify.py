from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import argparse
import pandas as pd
import os

def tsne_func(X, y):
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)

    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--xlsx_path', required=False, default="/media/user/easystore/Arbeitskopie-Myriad HRD Fälle bis 30 12 2021.xlsx")
    args = parser.parse_args()

    with pd.ExcelFile(args.xlsx_path) as xlsx:
        # xlsx = pd.ExcelFile(args.csv)
        worksheet = pd.read_excel(xlsx, "codiert")

        for c, row in worksheet.iterrows():
            simclr_paths = row["simclr_results"]
            if pd.notna(simclr_paths):
                simclr_paths = simclr_paths.replace("[", "").replace("'", "").replace("]", "")

                for path in simclr_paths.split(","):
                    name = path.split("/results")[0].split("/")[-1]
                    filepath = os.path.join(path, name + ".h5")
                    if os.path.isfile(filepath):
                        print(filepath)
