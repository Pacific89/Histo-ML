import os
import shutil


if __name__ == "__main__":

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