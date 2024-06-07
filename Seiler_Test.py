import numpy as np
import torch
import cv2


def get_neighbors(curr_num, comp_points):
    """
    Gets the closest neighbors based on euclidean distance to curr_num
    """

    pos = np.load("patch_preselection/template_positions.npy")
    curr_pos = pos[curr_num]

    # Calculate the Euclidean distance
    dist = np.sqrt(np.sum(np.abs(curr_pos - pos)**2, axis=1))

    idx_sorted = np.argsort(dist, axis=0)[1:comp_points+1]

    return idx_sorted


def haversine_distance_to_neighbors(curr_num, comp_points):
    """
    Gets the closest neighbors based on haversine distance to curr_num
    """
    points = np.load("patch_preselection/template_positions.npy")
    point = points[curr_num]
    x1, y1, z1 = point
    x2, y2, z2 = points.T

    # Convert Cartesian coordinates to spherical coordinates (latitude and longitude)
    lat1 = np.arcsin(z1)
    lon1 = np.arctan2(y1, x1)
    lat2 = np.arcsin(z2)
    lon2 = np.arctan2(y2, x2)

    # Calculate haversine distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 2 * np.arcsin(np.sqrt(a))

    # Sort distances and get indices of nearest neighbors
    nearest_neighbor_indices = np.argsort(distances)

    return nearest_neighbor_indices[1:comp_points + 1]


if __name__ == '__main__':

    data = np.genfromtxt('results/zs6d_ycbv-test_exact.csv', delimiter=',')

    print(data)
