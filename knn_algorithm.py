import numpy as np
import reader as rd
import random
from typing import List, Tuple, Optional
import pickle
import time
from collections import Counter
from scipy.spatial import cKDTree

# -------------------------------------
# 1. Weight 기반 스케일링/역스케일링 (벡터화)
# -------------------------------------
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

def scale_points(data: np.ndarray, weight: List[float]) -> np.ndarray:
    weight_arr = np.array(weight, dtype=float)
    sqrt_weight = np.sqrt(weight_arr)
    return data * sqrt_weight

def invert_scale_points(scaled_data: np.ndarray, weight: List[float]) -> np.ndarray:
    weight_arr = np.array(weight, dtype=float)
    sqrt_weight = np.sqrt(weight_arr)
    return scaled_data / sqrt_weight

# -------------------------------------
# 2. Pre-indexing: cKDTree Forest 구축
# -------------------------------------
def build_all_indexes(data: np.ndarray, weight_list: List[List[float]]) -> List[cKDTree]:
    kd_tree_list = []
    for w in weight_list:
        scaled_data = scale_points(data, w)
        tree = cKDTree(scaled_data)
        kd_tree_list.append(tree)
    return kd_tree_list

# -------------------------------------
# 3. 가중치 벡터 비교 (코사인 유사도 기반)
# -------------------------------------
def cosine_similarity(w1: List[float], w2: List[float]) -> float:
    w1_arr = np.array(w1, dtype=float)
    w2_arr = np.array(w2, dtype=float)
    dot_val = np.dot(w1_arr, w2_arr)
    norm1 = np.linalg.norm(w1_arr)
    norm2 = np.linalg.norm(w2_arr)
    return dot_val / (norm1 * norm2)

def distance_in_weight_space(w1: List[float], w2: List[float]) -> float:
    return 1.0 - cosine_similarity(w1, w2)

# -------------------------------------
# 4. 저장 및 불러오기 (pickle)
# -------------------------------------
def save_forest(weight_list: List[List[float]], kd_tree_list: List[cKDTree], filename: str = 'forest.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((weight_list, kd_tree_list), f)
    print("Forest saved to", filename)

def load_forest(filename: str = 'forest.pkl') -> Tuple[Optional[List[List[float]]], Optional[List[cKDTree]]]:
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
def query_AkNN(q: List[float], w_user: List[float],
               weight_list: List[List[float]], kd_tree_list: List[cKDTree],
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

    # Scale query vector
    q_scaled = np.array(q, dtype=float) * np.sqrt(np.array(chosen_weight, dtype=float))
    distances, indices = chosen_tree.query(q_scaled, k=K)
    neighbors = []
    if K == 1:
        neighbors.append((distances, (chosen_tree.data[indices].tolist(), int(indices))))
    else:
        for d, idx in zip(distances, indices):
            neighbors.append((d, (chosen_tree.data[idx].tolist(), int(idx))))
    return neighbors, new_tree_made

# -------------------------------------
# 6. Query: Exact kNN 검색 (cKDTree 활용)
# -------------------------------------
def query_exactKNN(q: List[float], w_user: List[float], data: np.ndarray, K: int) -> List[tuple]:
    sqrt_w = np.sqrt(np.array(w_user, dtype=float))
    scaled_data = data * sqrt_w
    tree = cKDTree(scaled_data)
    q_scaled = np.array(q, dtype=float) * sqrt_w
    distances, indices = tree.query(q_scaled, k=K)
    neighbors = []
    if K == 1:
        neighbors.append((distances, (tree.data[indices].tolist(), int(indices))))
    else:
        for d, idx in zip(distances, indices):
            neighbors.append((d, (tree.data[idx].tolist(), int(idx))))
    return neighbors

# -------------------------------------
# 7. 평가 함수: 성능 및 정확도 측정
# -------------------------------------
def classify_knn(labels: List[int]) -> int:
    return Counter(labels).most_common(1)[0][0]

def measure_preindex_time(data: np.ndarray, weight_list: List[List[float]]) -> float:
    start = time.perf_counter()
    _ = build_all_indexes(data, weight_list)
    end = time.perf_counter()
    return end - start

def measure_loading_time(filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    _ = load_forest(filename)
    end = time.perf_counter()
    return end - start

def measure_saving_time(weight_list: List[List[float]], kd_tree_list: List[cKDTree], filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    save_forest(weight_list, kd_tree_list, filename)
    end = time.perf_counter()
    return end - start

def query_info() -> Tuple[np.ndarray, np.ndarray]:
    # dry_bean 데이터셋의 각 feature에 대한 통계량 (평균, 분산)
    data_np, _ = rd.read_dataset("dry_bean")
    means = np.mean(data_np, axis=0)
    variances = np.var(data_np, axis=0)
    std_devs = np.sqrt(variances)    
    # 평균 ± 3 * 표준편차 구간 내에서 범위 계산
    lower = means - 3 * std_devs
    upper = means + 3 * std_devs
    # 음수 값이 있으면 0으로 보정 (각 원소별 보정)
    lower = np.maximum(lower, 0)
            
    return lower, upper

def generate_query() -> Tuple[List[float], List[float]]:
    # 미리 계산된 각 feature의 lower, upper bound 사용
    lower, upper = query_info()
    dim = len(lower)
    
    # 각 차원별로 무작위 query 값 생성
    q = [random.uniform(l, u) for l, u in zip(lower, upper)]
    # 각 차원별 weight 생성 (0.01 ~ 1.0)
    w_user = [round(random.uniform(0.01, 1.0), 3) for _ in range(dim)]
    
    return q, w_user

def evaluate_queries(num_queries: int, labels: List[int], data: np.ndarray, 
                     weight_list: List[List[float]], kd_tree_list: List[cKDTree], K: int) -> Tuple[float, float, float, float, float, float]:
    
    total_query_time = 0.0
    total_exact_query_time = 0.0
    correct_class_count = 0
    total_query_error = 0.0
    total_exact_matches = 0
    total_new_tree_count = 0
    
    for _ in range(num_queries):
        # 랜덤 query와 user weight
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
        
        # (3-1) classification 정확도 (akNN 결과와 exact 결과의 다수결 비교)
        aknn_indices = [cand[1] for (_, cand) in aknn_results]
        exact_indices = [cand[1] for (_, cand) in exact_results]
        
        aknn_labels = [labels[idx] for idx in aknn_indices]
        exact_labels = [labels[idx] for idx in exact_indices]
        
        class_aknn = classify_knn(aknn_labels)
        class_exact = classify_knn(exact_labels)
        
        if class_aknn == class_exact:
            correct_class_count += 1
            
        # (3-2) 이웃 간 거리 오차: akNN 결과와 exact 결과의 차이(각 query당 K로 나눈 평균 오차)
        query_error = 0.0
        exact_points = data[exact_indices]
        for idx_aknn in aknn_indices:
            if idx_aknn not in exact_indices:
                aknn_point = data[idx_aknn]
                distances = np.linalg.norm(exact_points - aknn_point, axis=1)
                error = np.min(distances)
                query_error += error
        # query_error를 K로 나누어 평균 오차로 산출
        total_query_error += (query_error / K)
        
        # (3-3) 정확한 점의 개수 (교집합 크기)
        total_exact_matches += len(set(aknn_indices).intersection(exact_indices))

    avg_query_time = total_query_time / num_queries
    avg_exact_query_time = total_exact_query_time / num_queries
    classification_accuracy = correct_class_count / num_queries
    avg_query_error = total_query_error / num_queries
    avg_exact_matches = total_exact_matches / num_queries
    
    return avg_query_time, avg_exact_query_time, classification_accuracy, avg_query_error, avg_exact_matches, total_new_tree_count

# -------------------------------------
# 8. 메인: Reader API를 사용하여 Dry Bean 데이터셋 평가
# -------------------------------------
if __name__ == "__main__":
    random.seed(42)
    
    # 데이터셋 로드 (Reader API)
    data_np, labels_np = rd.read_dataset("dry_bean")
    data = data_np.astype(float)  # (n, d) numpy array
    labels = labels_np.tolist()    # label array (평가에 사용할 수 있음)
    
    n, dim = data.shape
    print(f"Dataset loaded with {n} points and {dim} features.")

    # Pre-indexing: 기존 forest 불러오기; 없으면 새로 생성
    weight_list, kd_tree_list = load_forest("forest.pkl")
    if weight_list is None or kd_tree_list is None:
        var = np.var(data, axis=0)
        max_var = np.max(var)
        weight_list = []
        num_initial_weights = 20  # 생성할 weight vector 개수

        for _ in range(num_initial_weights):
            weight = []
            for i in range(dim):
                # 분산이 0일 때는 아주 좁은 구간, 분산이 최대일 때는 구간 폭이 1에 가까워짐.
                r_i = 0.1 + 0.9 * (var[i] / max_var)  # 구간 폭 결정
                lower_bound = max(0, 0.5 - r_i / 2)
                upper_bound = min(1, 0.5 + r_i / 2)
                w = round(random.uniform(lower_bound, upper_bound), 3)
                weight.append(float(w))
            weight_list.append(weight)
        
    kd_tree_list = build_all_indexes(data, weight_list)
    save_forest(weight_list, kd_tree_list, "forest.pkl")
    
    # 예시 출력
    for i, w in enumerate(weight_list):
        print(f"Tree {i+1}: weight = {w}")
    
    random.seed()
    
    # 1번: 사전 인덱싱 시간 측정
    loading_time = measure_loading_time("forest.pkl")
    print(f"Loading time for forest: {loading_time:.4f} seconds.")
    
    preindex_time = measure_preindex_time(data, weight_list)
    print(f"Pre-indexing time for forest: {preindex_time:.4f} seconds.")

    # 2번: 1000번 random query에 대해 akNN 검색 시간 측정 (Exact 제외)
    num_queries = 1000
    K = 6
    avg_q_time, avg_exact_q_time, class_acc, avg_q_err, avg_exact_matches, new_tree_count = evaluate_queries(num_queries, labels, data, weight_list, kd_tree_list, K)
    print("\n=== Query Evaluation over {} queries ===".format(num_queries))
    print(f"Average akNN query time: {avg_q_time:.6f} seconds per query.")
    print(f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.")
    print(f"Classification accuracy (akNN vs exact): {class_acc:.4f}")
    print(f"Average distance error: {avg_q_err:.4f}")
    print(f"Average exact neighbor matches: {avg_exact_matches:.2f} (out of {K})")
    print(f"{new_tree_count} new trees were created during the queries.")
    
    # 최종 forest 저장
    save_forest(weight_list, kd_tree_list, "forest.pkl")
