import numpy as np
from typing import Tuple

# 두 점 사이 거리 계산
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

# Weight 기반 스케일링/역스케일링
def scale_points(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return data * np.sqrt(weight.astype(float))

def invert_scale_points(scaled_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = weight.astype(float)
    sqrt_weight = np.sqrt(weight)
    safe_sqrt_weight = np.where(sqrt_weight == 0, 1, sqrt_weight)
    return scaled_data / safe_sqrt_weight

# Weight 간 유사도 계산
def distance_in_weight_space(w_user: np.ndarray, weight_list: np.ndarray) -> Tuple[int, float]:
    norm_w_user = np.linalg.norm(w_user)
    norms = np.linalg.norm(weight_list, axis=1)
    cosine_similarities = np.dot(weight_list, w_user) / (norms * norm_w_user)
    distances = 1.0 - cosine_similarities
    
    best_idx = np.argmin(distances)
    best_metric = distances[best_idx]
    
    return best_idx, best_metric
