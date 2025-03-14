from typing import List, Tuple, Optional
import numpy as np
import cloudpickle
from scipy.spatial import cKDTree
from scaling import scale_points, distance_in_weight_space

# 주어진 weight_list와 data로부터 모든 트리를 생성
def build_all_indexes(data: np.ndarray, weight_list: np.ndarray) -> List[cKDTree]:
    sqrt_weights = np.sqrt(weight_list)
    scaled_data_all = data[None, :, :] * sqrt_weights[:, None, :]
    kd_tree_list = [cKDTree(scaled_data_all[i]) for i in range(scaled_data_all.shape[0])]
    return kd_tree_list

# 가중치 리스트와 kd_tree 리스트를 파일로 저장
def save_forest(weight_list: np.ndarray, kd_tree_list: List[cKDTree], filename: str = 'forest.pkl') -> None:
    with open(filename, 'wb') as f:
        cloudpickle.dump((weight_list, kd_tree_list), f)
    print("Forest saved to", filename)

# 파일로부터 가중치 리스트와 kd_tree 리스트를 불러옴
def load_forest(filename: str = 'forest.pkl') -> Tuple[Optional[np.ndarray], Optional[List[cKDTree]]]:
    try:
        with open(filename, 'rb') as f:
            weight_list, kd_tree_list = cloudpickle.load(f)
        print("Forest loaded from", filename)
        return weight_list, kd_tree_list
    except (FileNotFoundError, EOFError):
        print("No valid saved forest found. A new one will be created.")
        return None, None

# 가중치 리스트 출력
def check_forest():
    weight_list = load_forest('forest.pkl')[0]
    for i, w in enumerate(weight_list):
        print(f"Tree {i}: {w}")

# AKNN 알고리즘 구현
def query_AkNN(q: np.ndarray, w_user: np.ndarray,
               weight_list: List[np.ndarray], kd_tree_list: List[cKDTree],
               data: np.ndarray,
               threshold: float, K: int) -> Tuple[List[tuple], bool]:
    best_idx = -1
    best_metric = float('inf')
    
    for i, w in enumerate(weight_list):
        sim_dist = distance_in_weight_space(w_user, w)
        if sim_dist < best_metric:
            best_metric = sim_dist
            best_idx = i

    if best_metric <= threshold:
        chosen_weight = weight_list[best_idx]
        chosen_tree = kd_tree_list[best_idx]
        new_tree_made = False
    else:
        chosen_weight = w_user
        new_scaled_data = scale_points(data, w_user)
        new_tree = cKDTree(new_scaled_data)
        weight_list.append(w_user)
        kd_tree_list.append(new_tree)
        chosen_tree = new_tree
        new_tree_made = True

    q_scaled = q * np.sqrt(chosen_weight)
    distances, indices = chosen_tree.query(q_scaled, k=K)
    
    if K == 1:
        neighbors = [(distances, (chosen_tree.data[indices].tolist(), int(indices)))]
    else:
        neighbors = [(d, (chosen_tree.data[idx].tolist(), int(idx))) 
                     for d, idx in zip(distances, indices)]
    return neighbors, new_tree_made

# 정확한 KNN 알고리즘 구현
def query_exactKNN(q: np.ndarray, w_user: np.ndarray, data: np.ndarray, K: int) -> List[tuple]:
    sqrt_w = np.sqrt(w_user)
    scaled_data = data * sqrt_w
    tree = cKDTree(scaled_data)
    q_scaled = q * sqrt_w
    distances, indices = tree.query(q_scaled, k=K)
    
    if K == 1:
        neighbors = [(distances, (tree.data[indices].tolist(), int(indices)))]
    else:
        neighbors = [(d, (tree.data[idx].tolist(), int(idx))) for d, idx in zip(distances, indices)]
    return neighbors
