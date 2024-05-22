import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm


def get_neighbors_with_dist(curr_num, comp_points):
    """
    Gets the closest neighbors based on the Euclidean distance to curr_num
    """
    pos = np.load("./template_positions.npy")
    curr_pos = pos[curr_num]

    # Calculate the Euclidean distance
    dist = np.sqrt(np.sum(np.abs(curr_pos - pos) ** 2, axis=1))

    idx_sorted = np.argsort(dist, axis=0)[1:comp_points + 1]

    return idx_sorted, dist[idx_sorted]


def haversine_distance_to_neighbors(curr_num, comp_points):
    """
    Gets the closest neighbors based on haversine distance to curr_num
    """
    points = np.load("./template_positions.npy")
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
    idx_sorted = nearest_neighbor_indices[1:comp_points + 1]

    return idx_sorted, distances[idx_sorted]


def evaluate_preselect_patches(patches_path: str, obj_id: int, num_comp: int=32,
                               weight: str='1', plot: bool=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sorted_idx_temp = []

    for template_id in tqdm.tqdm(range(642)):

        sim_matrix = torch.load(patches_path + f'obj_{obj_id}/sim_{template_id}.pt')
        patches_histogram = np.zeros(shape=(361, 1))

        nn_list, dist = haversine_distance_to_neighbors(template_id, num_comp)

        # Evaluate the best Patches that lead to BB:
        for k in range(num_comp):
            # Plot the Cosine Similarity over two templates
            sim_matrix_k = sim_matrix[k, :, :, :]
            if weight == '1':
                scaling_factor = 1
            elif weight == '1/dist':
                scaling_factor = 1 / dist[k]
            elif weight == 'exp':
                scaling_factor = np.exp(-dist[k])
            else:
                print(f"Invalid weight option: {weight} => scaling_factor = 1")
                scaling_factor = 1

            # Get BB Mask
            image_idxs = torch.arange(361, device=device)
            sim_1, nn_1 = torch.max(sim_matrix_k, dim=-1)# nn_1 - indices of block2 closest to block1
            sim_2, nn_2 = torch.max(sim_matrix_k, dim=-2)# nn_2 - indices of block1 closest to block2
            sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
            sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
            bbs_mask = nn_2[nn_1] == image_idxs

            # Add BB Histogram with weights
            patches_histogram[bbs_mask.detach().cpu().numpy()] += 1*scaling_factor

        # Plot the Resulting 19x19 Histogram
        if plot:
            patches_histogram_2d = patches_histogram.reshape((19, 19))
            plt.imshow(patches_histogram_2d, cmap='viridis', origin='lower', vmin=0,
                       vmax=num_patches)
            cbar = plt.colorbar(ticks=[0, num_patches / 2, num_patches])
            cbar.ax.tick_params(labelsize=22)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Histogram, n={num_patches}, w={weight}, '
                       f'T={obj_id}_{template_id}')
            plt.show()

        sorted_idx_temp.append(np.argsort(patches_histogram.T))

    ranked_patches = np.array(sorted_idx_temp).reshape(361, 642)
    np.save(f"./obj_{obj_id}_ranked_patches.npy", ranked_patches)
    return ranked_patches


if __name__ == '__main__':

    # Parameters
    patches_path = './ycbv_patches/'
    num_patches = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = ['1', '1/dist', 'exp']
    weight = weights[2]
    plot = True
    plot_k = False

    for obj_id in range(1, 22):
        patches = evaluate_preselect_patches(patches_path, obj_id, num_patches, weights[0], plot=False)

    """
    object_id = 1
    template_id = 427

    sim_matrix = torch.load(patches_path + f'obj_{object_id}/sim_{template_id}.pt')
    res_matrix = np.zeros(shape=(sim_matrix.shape[3], sim_matrix.shape[4]))
    patches_histogram = np.zeros(shape=(361, 1))

    nn_list, dist = haversine_distance_to_neighbors(template_id, num_patches)
    print(f"NN: {nn_list}, Distance: {dist}")

    # Evaluate the best Patches that lead to BB:
    for k in range(num_patches):
        # Plot the Cosine Similarity over two templates
        sim_matrix_k = sim_matrix[k, :, :, :]
        sim_array = sim_matrix_k.squeeze().detach().cpu().numpy()
        if weight == '1':
            scaling_factor = 1
        elif weight == '1/dist':
            scaling_factor = 1/dist[k]
        elif weight == 'exp':
            scaling_factor = np.exp(-dist[k])
        else:
            print(f"Invalid weight option: {weight} => scaling_factor = 1")
            scaling_factor = 1

        if k in [3, 7, 15, 31] and plot_k:
            plt.imshow(res_matrix/(np.max(res_matrix)), cmap='viridis', origin='lower', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Average Cosine Similarity @k={k+1}")
            plt.axis('off')
            plt.show()

        # Get BB Mask
        image_idxs = torch.arange(361, device=device)
        # nn_1 - indices of block2 closest to block1
        sim_1, nn_1 = torch.max(sim_matrix_k, dim=-1)
        # nn_2 - indices of block1 closest to block2
        sim_2, nn_2 = torch.max(sim_matrix_k, dim=-2)
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
        bbs_mask = nn_2[nn_1] == image_idxs

        patches_histogram[bbs_mask.detach().cpu().numpy()] += 1*scaling_factor

    if plot:
        patches_histogram_2d = patches_histogram.reshape((19, 19))
        patches_histogram_2d *= 32/patches_histogram.max()
        plt.imshow(patches_histogram_2d, cmap='viridis', origin='lower', vmin=0, vmax=num_patches)
        cbar = plt.colorbar(ticks=[0, num_patches/2, num_patches])
        cbar.ax.tick_params(labelsize=22)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(f'Histogram, n={num_patches}, w={weight}, T={object_id}_{template_id}')
        plt.show()

"""



