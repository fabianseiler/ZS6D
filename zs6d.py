import os
import json
import torch
from tqdm import tqdm
import numpy as np
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import logging
from src.pose_extractor import PoseViTExtractor
import refinement.refinement as ref


class ZS6D:

    def __init__(self, templates_gt_path, norm_factors_path, model_type='dino_vits8', stride=4, subset_templates=1, max_crop_size=80):
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size

        # Added by Fabian Seiler for refinement, TODO: Add parameter
        self.obj_poses = np.load("refinement/obj_poses.npy")
        self.refinement = "none"    #'ctr'

        try:
            with open(os.path.join(templates_gt_path), 'r') as f:
                self.templates_gt = json.load(f)

            with open(os.path.join(norm_factors_path), 'r') as f:
                self.norm_factors = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load templates or norm_factors: {e}")
            raise

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device={self.device}")

        self.extractor = PoseViTExtractor(model_type=self.model_type, stride=self.stride, device=self.device)

        self.templates_desc = {}
        templates_gt_subset = {}
        try:
            for obj_id, template_labels in tqdm(self.templates_gt.items()):
                self.templates_desc[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0)
                                                    for i, template_label in enumerate(template_labels)
                                                    if i % subset_templates == 0], dim=0)
                
                templates_gt_subset[obj_id] = [template_label for i, template_label in
                                            enumerate(template_labels) if i % subset_templates == 0]
                
        except Exception as e:
            self.logger.error(f"Error processing template descriptors: {e}")
            raise

        self.templates_gt = templates_gt_subset

        self.logger.info("Preparing templates and loading of extractor is done!")

    def get_pose(self, img, obj_id, mask, cam_K, num_comp, bbox=None):
        try:
            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            # Get Preprocessed Image
            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor.preprocess(img_crop, load_size=224)

            with torch.no_grad():
                desc = self.extractor.extract_descriptors(img_prep.to(self.device), layer=11, facet='key', bin=False, include_cls=True)
                desc = desc.squeeze(0).squeeze(0).detach().cpu()

            # Match to best fitting Template
            matched_templates = utils.find_template_cpu(desc, self.templates_desc[obj_id], num_results=1)
            print(f"Matches Template: {matched_templates[0][1]}")

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            with torch.no_grad():
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0]
                else:
                    crop_size = self.max_crop_size

                resize_factor = float(crop_size) / img_crop.size[0]

                # Local correspondence matching (Comparison btw Key(p_i) & Key(q_j))
                points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans(img_crop, template, num_pairs=20, load_size=crop_size)


                # TODO: Patch Preselection Implementation
                template_id = matched_templates[0][1]

                #points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_preselect(obj_id,
                #                                                                                         template_id,
                #                                                                                         num_comp,
                #                                                                                         img_crop,
                #                                                                                         template,
                #                                                                                         num_pairs=20,
                #                                                                                         load_size=crop_size)

                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)
                img_uv = cv2.resize(img_uv, (crop_size, crop_size))

                # Pose estimation from valid correspondences between Template-Patches and Image-Patches
                R_est, t_est = utils.get_pose_from_correspondences(points1, points2, 
                                                                   y_offset, x_offset, 
                                                                   img_uv, cam_K, 
                                                                   self.norm_factors[str(obj_id)], 
                                                                   scale_factor=1.0, 
                                                                   resize_factor=resize_factor)

                # TODO: Revert Changes when BOP evaluation is fixed
                # Refinement Stage: Closest Template Refinement of Not
                if self.refinement == 'ctr':
                    R_ref, t_ref = self.refine_pose_closest_template(R_est, t_est, obj_id, img_crop, crop_size,
                                                                     y_offset, x_offset, cam_K, resize_factor)
                else:
                    R_ref, t_ref = False, False

                return R_est, t_est
        except Exception as e:
            self.logger.error(f"Error in get_pose: {e}")
            raise

    def refine_pose_closest_template(self, R_est, t_est, obj_id, img_crop,
                                     crop_size, y_offset, x_offset, cam_K,
                                     resize_factor):
        """ Refinement based on the R_est Matrix that finds the closest Template
        to the guessed R matrix and recalculates LCM """
        # Load all Object Poses
        R_t = self.obj_poses[:, :3, :3]
        # Get Template ID for the Template with the most similar Rot Matrix
        temp_idx = ref.compare_templates(R_est, R_t)
        #temp_idx = ref.compare_templates_cos(R_est, R_t)
        print(f"Refinement Template: {temp_idx}")

        # Read in new Template
        template_ref = Image.open(self.templates_gt[obj_id][temp_idx]['img_crop'])

        # Recalculate LCM:
        points1, points2, crop_pil, template_pil = self.extractor.find_correspondences_fastkmeans(img_crop, template_ref, num_pairs=20, load_size=crop_size)
        if not points1 or not points2:
            raise ValueError("Insufficient correspondences found @Refinement.")

        # Load Template as numpy array and resize it accordingly
        img_uv = np.load(f"{self.templates_gt[obj_id][temp_idx]['img_crop'].split('.png')[0]}_uv.npy")
        img_uv = img_uv.astype(np.uint8)
        img_uv = cv2.resize(img_uv, (crop_size, crop_size))

        # Redo the Pose estimation with the new Template
        R_ref, t_ref = utils.get_pose_from_correspondences(points1, points2,
                                                           y_offset, x_offset,
                                                           img_uv, cam_K,
                                                           self.norm_factors[
                                                               str(obj_id)],
                                                           scale_factor=1.0,
                                                           resize_factor=resize_factor)
        return R_ref, t_ref

    """
    def get_all_desc(self, obj_id):
        return torch.cat([torch.from_numpy(np.load(f"{self.template_imgs_path}obj_{obj_id}/" + "{:06d}.npy".format(i))).unsqueeze(0)
                         for i in range(642)], dim=0)

    """

    """
    def refine_pose_foundpose(self, R_est, t_est,
                              crop_size, img_crop):

        # Get feature map F_q by stacking the Descriptors
        image_batch, image_pil, scale_factor = self.extractor.preprocess(img_crop, crop_size)
        descriptors = self.extractor.extract_descriptors(image_batch.to(self.device), layer=9, facet='key', bin=True)

        F_q = descriptors   # Check how to stack, R**(axaxd) with axa = #Patches

        
        # R,t are refined by Levenberg-Marquardt (iterative/non-linear)
        # TODO: Find functionality of LM optimization

        # Updated Error Metric by minimizing the Barron Cost function

        return  # R_new, t_new
        
    """