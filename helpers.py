



def get_clam_paths(project_name, parent_path):
    wsi_paths = []
    blockmap_paths = []
    wsi_names = []
    print("Getting Files for Project: ", project_name)

    for root, dirs, files in os.walk(parent_path):
        # print(root)
        if "data" in root:
            if len(files) == 1:
                wsi_path = os.path.join(root, files[0])
                # print(wsi_path)
                wsi_paths.append(wsi_path)

        for subfolder in dirs:
            if subfolder == "production":
                img_dir = os.path.join(root, "production/heat/Unspecified")
                
                for img in os.listdir(img_dir):
                    if "orig_2" in img:
                        orig_path = os.path.join(img_dir, img)
                        # print("Original: ", orig_path)
                    else:
                        heat_path = os.path.join(img_dir, img)
                        # print("Heat: ", heat_path)

                name = root.split(project_name + "/")[1].split("/")[0]
                wsi_names.append(name)
                # print(name)

            elif subfolder == "raw":
                blockmap_file = os.path.join(root, "raw/heat/Unspecified/{0}/{0}_blockmap.h5".format(name))
                # print("Block:", blockmap_file)
                if os.path.isfile(blockmap_file):
                    blockmap_paths.append(blockmap_file)
                else:
                    blockmap_paths.append("missing")


    print("Total Files: ", len(wsi_names))
    wsi_dict = {}
    missing_blockmaps = []
    for ind, block in enumerate(blockmap_paths):
        if not block == "missing":
            # print("Missing Blockmap: ", wsi_names[ind])
            wsi_dict.update({wsi_names[ind]: [wsi_paths[ind], blockmap_paths[ind]]})
        else:
            missing_blockmaps.append(wsi_names[ind])

    if len(wsi_names) == len(list(wsi_dict.keys())):
        print("All Blockmaps Available for project {0}".format(self.project_name))
    else:
        print("Blockmaps Missing for:", missing_blockmaps)

    print("Proceeding with {0} files".format(len(list(wsi_dict.keys()))))

    return wsi_dict