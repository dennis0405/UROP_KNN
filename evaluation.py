import time
from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree
from kd_tree_forest import build_all_indexes, save_forest, load_forest
from kd_tree_forest import query_AkNN, query_exactKNN
from classification import classify_knn_weighted

# kd forest를 만드는 데 걸리는 시간 측정
def measure_preindex_time(data: np.ndarray, weight_list: np.ndarray) -> float:
    start = time.perf_counter()
    kd_tree_list = build_all_indexes(data, weight_list)
    end = time.perf_counter()
    return end - start, kd_tree_list

# kd forest를 불러오는 데 걸리는 시간 측정
def measure_loading_time(filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    _ = load_forest(filename)
    end = time.perf_counter()
    return end - start

# kd forest를 저장하는 데 걸리는 시간 측정
def measure_saving_time(weight_list: np.ndarray, kd_tree_list: List[cKDTree], filename: str = 'forest.pkl') -> float:
    start = time.perf_counter()
    save_forest(weight_list, kd_tree_list, filename)
    end = time.perf_counter()
    return end - start

# query 생성
def generate_query(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.round(np.random.uniform(-3, 3, size=dim), 3)
    w_user = np.round(np.random.uniform(0, 1, size=dim), 3)
    return q, w_user

# query 평가
def evaluate_queries(num_queries: int, labels: List[str], data: np.ndarray, 
                     weight_list: np.ndarray, kd_tree_list: List[cKDTree], K: int, threshold: float) -> Tuple[float, float, float, float, float, float]:
    
    total_query_time = 0.0
    total_exact_query_time = 0.0
    correct_class_count = 0
    total_query_error = 0.0
    total_exact_matches = 0
    total_new_tree_count = 0
    dim = data.shape[1]
    
    for _ in range(num_queries):
        q, w_user = generate_query(dim)
        
        start = time.perf_counter()
        aknn_results, new_tree_made = query_AkNN(q, w_user, weight_list, kd_tree_list, data, threshold, K=K)
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
        
        # error 산출 방식에 개선 필요
        query_error = 0.0
        exact_points = data[[cand[1] for (_, cand) in exact_results]]
        exact_indices = set([cand[1] for (_, cand) in exact_results])
        for _, cand in aknn_results:
            idx = cand[1]
            if idx not in exact_indices:
                aknn_point = data[idx]
                distances = np.linalg.norm(exact_points - aknn_point, axis=1)
                error = np.min(distances)
                query_error += error
        total_query_error += (query_error / K)
        
        aknn_indices = set([cand[1] for (_, cand) in aknn_results])
        total_exact_matches += len(aknn_indices.intersection(exact_indices))
    
    avg_query_time = total_query_time / num_queries
    avg_exact_query_time = total_exact_query_time / num_queries
    classification_accuracy = correct_class_count / num_queries
    avg_query_error = total_query_error / num_queries
    avg_exact_matches = total_exact_matches / num_queries
    
    return avg_query_time, avg_exact_query_time, classification_accuracy, avg_query_error, avg_exact_matches, total_new_tree_count
