import numpy as np
import pandas as pd
import random
import pickle
import time
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import RobustScaler

data_means = None
data_stds = None
dim = None

# -------------------------------------
# 1. Weight 기반 스케일링/역스케일링 (벡터화)
# -------------------------------------
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

def scale_points(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return data * np.sqrt(weight.astype(float))

def invert_scale_points(scaled_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = weight.astype(float)
    sqrt_weight = np.sqrt(weight)
    safe_sqrt_weight = np.where(sqrt_weight == 0, 1, sqrt_weight)
    return scaled_data / safe_sqrt_weight

# -------------------------------------
# 2. Pre-indexing: cKDTree Forest 구축
# -------------------------------------
def build_all_indexes(data: np.ndarray, weight_list: np.ndarray) -> List[cKDTree]:
    sqrt_weights = np.sqrt(weight_list)
    scaled_data_all = data[None, :, :] * sqrt_weights[:, None, :]
    kd_tree_list = [cKDTree(scaled_data_all[i]) for i in range(scaled_data_all.shape[0])]
    return kd_tree_list

# -------------------------------------
# 3. 가중치 벡터 비교 (코사인 유사도 기반)
# -------------------------------------
def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    dot_val = np.dot(w1, w2)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    return dot_val / (norm1 * norm2)

def distance_in_weight_space(w1: np.ndarray, w2: np.ndarray) -> float:
    return 1.0 - cosine_similarity(w1, w2)

# -------------------------------------
# 4. 저장 및 불러오기 (pickle)
# -------------------------------------
def save_forest(weight_list: np.ndarray, kd_tree_list: List[cKDTree], filename: str = 'forest.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((weight_list, kd_tree_list), f)
    print("Forest saved to", filename)

def load_forest(filename: str = 'forest.pkl') -> Tuple[Optional[np.ndarray], Optional[List[cKDTree]]]:
    try:
        with open(filename, 'rb') as f:
            weight_list, kd_tree_list = pickle.load(f)
        print("Forest loaded from", filename)
        return weight_list, kd_tree_list
    except FileNotFoundError:
        print("No saved forest found. A new one will be created.")
        return None, None

# -------------------------------------
# 5. Query: AkNN 검색 (cKDTree 기반, Threshold 적용)
# -------------------------------------
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

# -------------------------------------
# 6. Query: Exact kNN 검색 (cKDTree 활용)
# -------------------------------------
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

# -------------------------------------
# 7. Classification: weighted voting based on distance
# -------------------------------------
def classify_knn_weighted(neighbors: List[tuple], labels: List[int]) -> int:
    epsilon = 1e-6
    vote_dict = {}
    for dist, neighbor in neighbors:
        idx = neighbor[1]  # neighbor: (point, index)
        label = labels[idx]
        weight = 1.0 / (dist + epsilon)
        vote_dict[label] = vote_dict.get(label, 0) + weight
    return max(vote_dict.items(), key=lambda x: x[1])[0]

# -------------------------------------
# 8. 평가 함수: 성능 및 정확도 측정
# -------------------------------------
def measure_preindex_time(data: np.ndarray, weight_list: np.ndarray) -> float:
    start = time.perf_counter()
    kd_tree_list = build_all_indexes(data, weight_list)
    end = time.perf_counter()
    return end - start, kd_tree_list

def measure_loading_time(filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    _ = load_forest(filename)
    end = time.perf_counter()
    return end - start

def measure_saving_time(weight_list: np.ndarray, kd_tree_list: List[cKDTree], filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    save_forest(weight_list, kd_tree_list, filename)
    end = time.perf_counter()
    return end - start

def generate_query() -> Tuple[np.ndarray, np.ndarray]:
    global data_means, data_stds, dim
    lower = data_means - 3 * data_stds
    upper = data_means + 3 * data_stds
    q = np.random.uniform(lower, upper)
    w_user = np.round(np.random.uniform(0, 1, size=dim), 3)
    return q, w_user

def evaluate_queries(num_queries: int, labels: List[int], data: np.ndarray, 
                     weight_list: np.ndarray, kd_tree_list: List[cKDTree], K: int) -> Tuple[float, float, float, float, float, float]:
    
    total_query_time = 0.0
    total_exact_query_time = 0.0
    correct_class_count = 0
    total_query_error = 0.0
    total_exact_matches = 0
    total_new_tree_count = 0
    
    for _ in range(num_queries):
        q, w_user = generate_query()
        
        start = time.perf_counter()
        aknn_results, new_tree_made = query_AkNN(q, w_user, weight_list, kd_tree_list, data, threshold=0.12, K=K)
        end = time.perf_counter()
        total_query_time += (end - start)
        
        if new_tree_made:
            total_new_tree_count += 1
        
        start = time.perf_counter()
        exact_results = query_exactKNN(q, w_user, data, K=K)
        end = time.perf_counter()
        total_exact_query_time += (end - start)
        
        class_aknn = classify_knn_weighted(aknn_results, labels)
        class_exact = classify_knn_weighted(exact_results, labels)
        
        if class_aknn == class_exact:
            correct_class_count += 1
            
        query_error = 0.0
        exact_points = data[[cand[1] for (_, cand) in exact_results]]
        exact_indices_set = set([cand[1] for (_, cand) in exact_results])
        for _, cand in aknn_results:
            idx = cand[1]
            if idx not in exact_indices_set:
                aknn_point = data[idx]
                distances = np.linalg.norm(exact_points - aknn_point, axis=1)
                error = np.min(distances)
                query_error += error
        total_query_error += (query_error / K)
        
        aknn_indices = set([cand[1] for (_, cand) in aknn_results])
        exact_indices = set([cand[1] for (_, cand) in exact_results])
        total_exact_matches += len(aknn_indices.intersection(exact_indices))
    
    avg_query_time = total_query_time / num_queries
    avg_exact_query_time = total_exact_query_time / num_queries
    classification_accuracy = correct_class_count / num_queries
    avg_query_error = total_query_error / num_queries
    avg_exact_matches = total_exact_matches / num_queries
    
    return avg_query_time, avg_exact_query_time, classification_accuracy, avg_query_error, avg_exact_matches, total_new_tree_count

# -------------------------------------
# 9. 메인: scikit-learn의 kddcup99 데이터를 사용한 kNN 평가 (numeric feature만 사용)
# -------------------------------------
if __name__ == "__main__":
    random.seed()
    
    # kdd_cup 데이터셋 로드 (범주형 feature 인덱스 1, 2, 3 제거)
    data_np, labels_np = fetch_kddcup99(download_if_missing=True, return_X_y=True)
    print(f"Original kdd_cup Data shape: {data_np.shape}")
    
    df = pd.DataFrame(data_np)
    df_numeric = df.drop(columns=[1, 2, 3])
    data_np_numeric = df_numeric.astype(float).to_numpy()
    
    # RobustScaler를 사용해 데이터 표준화
    scaler = RobustScaler()
    data = scaler.fit_transform(data_np_numeric)
    
    # 라벨 변환 (bytes -> str)
    labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels_np]
    
    n, dim = data.shape
    print(f"Dataset loaded with {n} points and {dim} features (numeric only, standardized using RobustScaler).")
    
    data_means = np.mean(data, axis=0)
    data_stds = np.std(data, axis=0)
    
    weight_list, kd_tree_list = load_forest("forest.pkl")
    if weight_list is None or kd_tree_list is None:
        weight_list = []
        num_initial_weights = 20
        for _ in range(num_initial_weights):
            weight = np.round(np.random.uniform(0, 1, size=dim), 3).tolist()
            weight_list.append(weight)
    
    for i, w in enumerate(weight_list):
        print(f"Tree {i+1}: weight = {w}")
    
    preindex_time, kd_tree_list = measure_preindex_time(data, weight_list)
    print(f"Pre-indexing time for forest: {preindex_time:.4f} seconds.")
    
    saving_time = measure_saving_time(weight_list, kd_tree_list, "forest.pkl")
    print(f"Saving time for forest: {saving_time:.4f} seconds.")
    
    loading_time = measure_loading_time("forest.pkl")
    print(f"Loading time for forest: {loading_time:.4f} seconds.")
    
    num_queries = 100
    K = 20
    avg_q_time, avg_exact_q_time, class_acc, avg_q_err, avg_exact_matches, new_tree_count = evaluate_queries(num_queries, labels, data, weight_list, kd_tree_list, K)
    print("\n=== Query Evaluation over {} queries ===".format(num_queries))
    print(f"Average akNN query time: {avg_q_time:.6f} seconds per query.")
    print(f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.")
    print(f"Classification accuracy (akNN vs exact): {class_acc:.4f}")
    print(f"Average distance error: {avg_q_err:.4f}")
    print(f"Average exact neighbor matches: {avg_exact_matches:.2f} (out of {K})")
    print(f"{new_tree_count} new trees were created during the queries.")
    
    save_forest(weight_list, kd_tree_list, "forest.pkl")
