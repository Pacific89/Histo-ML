import os
import shutil


if __name__ == "__main__":
    """ files stored in "input_folder" are restructured such that every svs is stored inside a folder called "name/data/name.svs"
    E.G.:
    files stored in the folder "dataset":  "dataset/wsi-x.svs ; dataset/wsi-y.svs"
    are restructured in the following way: "dataset/wsi-x/data/wsi-x.svs ; dataset/wsi-y/data/wsi-y.svs etc" from
       
    """

    ext = ".svs"
    input_folder = "/media/user/easystore/HRD-Subset-V"

    dirlist = []

    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if f.endswith(ext):
                # dirlist.append(os.path.join())
                new_dir = f.split(".")[0]
                new_path = os.path.join(root, new_dir, "data")
                os.makedirs(new_path)
                curren_file = os.path.join(root, f)

                # print("moving: {0} to {1}".format(curren_file, new_path))
                shutil.move(curren_file, new_path)

    # [print(d) for d in dirlist]