import h5py
import pandas as pd

p_ = "/home/user/Documents/Master/data/one/data/clam/features/h5_files/one.h5"
keys = ['coords', 'features']

temp_list = []
with h5py.File(p_, 'r') as f:
    num_patches = f['coords'].shape[0]
    print(len(f.keys()))
    # for patch_num in range(num_patches):
        # patch_id = "patch_{}".format(patch_num)
        # d_ = {'patch_id' : patch_id, 'coord_x' : f['coords'][patch_num][0], 'coords_y' : f['coords'][patch_num][1], 'features' :f['features'][patch_num]}
        # print(d_)
        # temp_list.append(d_)

# df = pd.DataFrame.from_dict(temp_list).set_index('patch_id')
# df.to_json("/home/user/Documents/Master/results_clam.json", orient='index')