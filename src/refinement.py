import numpy as np
import numpy.linalg.linalg
from robust_loss_pytorch import *
from src.levenberg_marquardt import lm


def refine_surf_emb(self):
    # Render Current Pos Estimate
    # Project to 2D Image
    # Biliniar Interpolation
    return  # R_new, t_new

def refine_pose_megapose(self):

    # Run refinement function n times

    # Score Refinements

    return  # R_new, t_new


def refinement():
    return

def score_refinement():
    return

# ----------------------- Compare R Matrix function -------------------------- #


def compare_templates(R_est, t_est, R_t):
    """
    Compares the estimated Rot matrix to the Rot matricies of the Templates
    """
    # Read in the Rot Matrix for all templates
    aa_est = rot_matrix_to_angle_axis(R_est)

    # Get the angle axis representation for all rot matricies
    aa_t_list = [rot_matrix_to_angle_axis(rot) for rot in R_t]
    aa_t_array = tuple([np.array([v[0] for v in aa_t_list]),
                        np.array([v[1] for v in aa_t_list])])

    # Get the closest Template idx to R_est
    diff = angle_difference(aa_est, aa_t_array)
    sorted_idx = np.argsort(diff)[0]

    return sorted_idx


def rot_matrix_to_angle_axis(rotation_matrix):
    """
    Transforms a Rot Matrix into the angle axis representation
    """
    trace = np.trace(rotation_matrix)
    theta = np.arccos(np.clip(((trace - 1) / 2), -1, 1))
    if np.isclose(theta, 0):
        # No rotation, return arbitrary axis and angle 0
        return np.array([0, 0, 1]), 0
    else:
        v = 1 / (2 * np.sin(theta)) * np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                                                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                                                rotation_matrix[1, 0] - rotation_matrix[0, 1]])
        if np.sum(v) == 0:
            return v, theta
        return v / np.linalg.norm(v), theta


def angle_difference(angle_axis1, angle_axis_array):
    """
    Calculates the difference of R_est to all R_t in angle axis representation
    """

    axis1, angle1 = angle_axis1
    axes2, angles2 = angle_axis_array

    dot_products = np.dot(axes2, axis1)
    # Ensure dot products are within valid range for arccos
    cos_theta = np.clip(dot_products, -1, 1)

    angle_differences = np.arccos(cos_theta)

    # Adjust angle differences for opposite direction rotations
    opposite_direction_mask = np.isclose(np.abs(dot_products), 1.0)
    angle_differences[opposite_direction_mask] = np.pi

    print(angle_differences)

    return angle_differences


def compare_templates_cos(R_est, R_t):
    R_est_f = R_est.reshape(-1)
    R_t_f = R_t.reshape(R_t.shape[0], -1)
    sim = np.dot(R_t_f, R_est_f)
    return np.argsort(sim)[0]



if __name__ == '__main__':

    R_est = np.array([[-0.85133979, 0.52450129, -0.01090679],
                      [0.10908387, 0.15664633, -0.98161226],
                      [-0.51314839, -0.83687533, -0.19057388]])


    data = np.load("../templates/obj_poses.npy")[:, :3, :3]
    x = compare_templates_cos(R_est, data)
    print("fin")
