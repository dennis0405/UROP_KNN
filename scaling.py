import numpy as np

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
def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    dot_val = np.dot(w1, w2)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    return dot_val / (norm1 * norm2)

def distance_in_weight_space(w1: np.ndarray, w2: np.ndarray) -> float:
    return 1.0 - cosine_similarity(w1, w2)