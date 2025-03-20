import time
from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
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
def evaluate_queries(num_queries: int, labels: np.ndarray, data: np.ndarray, 
                     weight_list: np.ndarray, kd_tree_list: List[cKDTree], 
                     tree_usage: List[int], K: int, threshold: float, max_trees: int) -> Tuple[float, float, float, float, float, float]:
    
    query_time_list = []
    exact_query_time_list = []
    correct_list = []         
    query_error_list = []
    exact_matches_list_count = []
    new_tree_count_list = []
    
    dim = data.shape[1]
    
    for _ in range(num_queries):
        q, w_user = generate_query(dim)
        
        start = time.perf_counter()
        aknn_results, new_tree_made = query_AkNN(q, w_user, weight_list, kd_tree_list, data, tree_usage, threshold, K, max_trees)
        end = time.perf_counter()
        query_time_list.append(end - start)
        query_time_list.append(1 if new_tree_made else 0)
        
        start = time.perf_counter()
        exact_results = query_exactKNN(q, w_user, data, K)
        end = time.perf_counter()
        exact_query_time_list.append(end - start)
        
        class_aknn = classify_knn_weighted(aknn_results, labels)
        class_exact = classify_knn_weighted(exact_results, labels)
        correct_list.append(1 if class_aknn == class_exact else 0)
        
        approx_indices = np.array([idx for (_, idx) in aknn_results])
        exact_indices = np.array([idx for (_, idx) in exact_results])
        common = np.intersect1d(approx_indices, exact_indices)
        
        if len(common) < K:
            unmatched_approx = approx_indices[~np.isin(approx_indices, common)]
            unmatched_exact = exact_indices[~np.isin(exact_indices, common)]
            
            approx_points = data[unmatched_approx]   # shape: (n_unmatched, d)
            exact_points = data[unmatched_exact]       # shape: (m_unmatched, d)
                
            diff = approx_points[:, np.newaxis, :] - exact_points[np.newaxis, :, :]
            cost_matrix = np.linalg.norm(diff, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            query_err = cost_matrix[row_ind, col_ind].sum()
        else:
            query_err = 0.0    
        query_error_list.append(query_err/K)
        
        exact_matches_list_count.append(len(common))
    
    avg_query_time = np.mean(query_time_list)
    avg_exact_query_time = np.mean(exact_query_time_list)
    classification_accuracy = np.mean(correct_list)
    avg_query_error = np.mean(query_error_list)
    avg_exact_matches = np.mean(exact_matches_list_count) / K
    total_new_tree_count = np.sum(new_tree_count_list)
    
    return avg_query_time, avg_exact_query_time, classification_accuracy, avg_query_error, avg_exact_matches, total_new_tree_count

