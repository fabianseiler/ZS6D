import src.pose_extractor
import torch
import os
import tqdm

# Required Extra functions from src.pose_extractor to copy:
# create_desc_of_templates
# create_desc
# pre_rank_template_patches
# get_neighbors


if __name__ == '__main__':

    folder = "./templates/ycbv_desc/"
    save_dir_desc = "../templates/ycbv_test/"
    save_dir_patches = "./ycbv_patches/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")

    extractor = src.pose_extractor.PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)

    # Create all descriptors
    if not os.path.exists(save_dir_desc):
        os.mkdir(save_dir_desc)
    extractor.create_desc_of_templates(folder=folder, save_path_dir=save_dir_desc)
    print("Descriptor Extraction at Layer=9 DONE")

    # Create Similarities for all
    with torch.no_grad():
        for obj in tqdm.tqdm(os.listdir(save_dir_desc)):
            if not os.path.exists(save_dir_patches):
                os.mkdir(save_dir_patches)
            extractor.pre_rank_template_patches(save_dir_desc + obj, save_dir=save_dir_patches+obj, comp_points=32)
            print(f"Similarities of {obj} calculated!")

    print("Evaluate Patches Script DONE")

