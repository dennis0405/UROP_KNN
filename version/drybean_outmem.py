import math
import numpy as np
import reader as rd
import random
import heapq
from typing import List, Tuple, Optional
import pickle

# KD tree 구성 요소
class KDNode:
    def __init__(self, point: List[float], axis: int, left: 'KDNode' = None, right: 'KDNode' = None, index: int = None):
        self.point = point      # 스케일된 feature vector
        self.axis = axis        # 분할에 사용한 차원
        self.left = left        # 왼쪽 서브트리
        self.right = right      # 오른쪽 서브트리
        self.index = index      # 원본 데이터의 인덱스

# KD-Tree 구축 (recursive)
def build_kd_tree(data_tuples: List[Tuple[List[float], int]]) -> Optional[KDNode]:
    # data_tuples: [(point 1, index 1), (point 2, index 2), ...]
    
    if not data_tuples:
        return None

    points = [pt for (pt, idx) in data_tuples]
    # points = [pt1, pt2, ...]
    # pt1 = [x1, x2, ...]
    
    points_np = np.array(points)
    var_per_axis = np.var(points_np, axis=0)
    axis = int(np.argmax(var_per_axis))
    # 분산이 가장 큰 axis 선택
    
    sorted_tuples = sorted(data_tuples, key=lambda x: x[0][axis])
    median_idx = len(sorted_tuples) // 2
    # axis 기준으로 오름차순 sort, 중간 값 선택

    median_point, median_index = sorted_tuples[median_idx]
    node = KDNode(point=median_point, axis=axis, index=median_index)
    
    left_tuples = sorted_tuples[:median_idx]
    right_tuples = sorted_tuples[median_idx+1:]
    
    node.left = build_kd_tree(left_tuples)
    node.right = build_kd_tree(right_tuples)
    
    return node

# 유클리드 distance
def distance(p1: List[float], p2: List[float]) -> float:
    return math.sqrt(sum((p1[i] - p2[i])**2 for i in range(len(p1))))

# k-d tree를 타고 내려가면서, nearest neighbor 찾는 함수 (recursive)
def _nn_search(node: KDNode, query: List[float], best_candidates: List[tuple], k: int):
    if node is None:
        return
    dist = distance(node.point, query)
    candidate = (node.point, node.index)
    
    if len(best_candidates) < k:
        heapq.heappush(best_candidates, (-dist, candidate))
    else:
        worst_dist = -best_candidates[0][0]
        if dist < worst_dist:
            heapq.heapreplace(best_candidates, (-dist, candidate))
    
    # 저장해놓은 axis 기준 분기
    axis = node.axis
    if query[axis] < node.point[axis]:
        next_branch = node.left
        other_branch = node.right
    else:
        next_branch = node.right
        other_branch = node.left
    _nn_search(next_branch, query, best_candidates, k)
    
    # k개의 neighbor을 찾지 못한 경우 or 다른 branch에도 가능성이 있을 때
    if len(best_candidates) < k or abs(query[axis] - node.point[axis]) < -best_candidates[0][0]:
        _nn_search(other_branch, query, best_candidates, k)

def kd_tree_knn_search(root: KDNode, query: List[float], k: int) -> List[tuple]:
    # knn 검색의 가장 바깥 함수
    # best candidates: [(distance, (point, index)), ...] -> min heap 관리
    
    best_candidates: List[tuple] = []
    _nn_search(root, query, best_candidates, k)
    result = [(-dist, (point, idx)) for (dist, (point, idx)) in best_candidates]
    result.reverse() # 기존의 minHeap을 reverse해서 정렬
    return result

# weight 기반 스케일링 및 역스케일링 함수
def scale_point(point: List[float], weight: List[float]) -> List[float]:
    point_arr = np.array(point, dtype=float)
    weight_arr = np.array(weight, dtype=float)
    sqrt_weight_arr = np.sqrt(weight_arr)
    scaled_arr = point_arr * sqrt_weight_arr
    return scaled_arr.tolist()

def invert_scale_point(scaled_point: List[float], weight: List[float]) -> List[float]:
    scaled_arr = np.array(scaled_point, dtype=float)
    weight_arr = np.array(weight, dtype=float)
    sqrt_weight_arr = np.sqrt(weight_arr)
    original_arr = scaled_arr / sqrt_weight_arr
    return original_arr.tolist()

# Pre-indexing: 여러 가중치에 대해 KD-Tree 생성 (인덱스 포함)
def build_all_indexes(indexed_data: List[Tuple[List[float], int]], weight_list: List[List[float]]) -> List[KDNode]:
    """
    indexed_data: [(feature, index), ...]
    weight_list: 여러 개의 weight 벡터 (각각 d차원)
    각 weight에 대해, 모든 feature를 스케일링한 후 kd-tree를 구축하여 리스트로 반환.
    """
    kd_tree_list = []
    for w in weight_list:
        scaled_indexed_data = []
        for (pt, idx) in indexed_data:
            scaled_pt = scale_point(pt, w)
            scaled_indexed_data.append((scaled_pt, idx))
        kd_tree = build_kd_tree(scaled_indexed_data)
        kd_tree_list.append(kd_tree)
    return kd_tree_list

# ========================
# 4. 가중치 벡터 비교 (코사인 유사도 기반)
# ========================
def cosine_similarity(w1: List[float], w2: List[float]) -> float:
    w1_arr = np.array(w1, dtype=float)
    w2_arr = np.array(w2, dtype=float)
    dot_val = np.dot(w1_arr, w2_arr)
    norm_w1 = np.linalg.norm(w1_arr)
    norm_w2 = np.linalg.norm(w2_arr)
    # 영벡터일 경우 예외처리 고려
    return dot_val / (norm_w1 * norm_w2)

def distance_in_weight_space(w1: List[float], w2: List[float]) -> float:
    return 1.0 - cosine_similarity(w1, w2)
    # 0~2 사이의 값 반환

# ========================
# 5. Weight Alterable AkNN 검색 (Threshold 적용)
# ========================
def query_AkNN(q: List[float], w_user: List[float],
               weight_list: List[List[float]], kd_tree_list: List[KDNode],
               indexed_data: List[Tuple[List[float], int]],
               threshold: float = 0.5, K: int = 3) -> Tuple[List[tuple], List[float]]:
    # return: (neighbors, chosen_weight)
    
    best_idx = -1
    best_metric = float('inf')
    for i, w in enumerate(weight_list):
        sim_dist = distance_in_weight_space(w_user, w)
        if sim_dist < best_metric:
            best_metric = sim_dist
            best_idx = i

    if best_metric < threshold:
        chosen_weight = weight_list[best_idx]
        chosen_tree = kd_tree_list[best_idx]
        print(f"Using existing weight index (distance {best_metric:.4f} < threshold {threshold}).")
    else:
        chosen_weight = w_user
        new_scaled_data = []
        for (pt, idx) in indexed_data:
            scaled_pt = scale_point(pt, w_user)
            new_scaled_data.append((scaled_pt, idx))
        new_tree = build_kd_tree(new_scaled_data)
        weight_list.append(w_user)
        kd_tree_list.append(new_tree)
        chosen_tree = new_tree
        print(f"New weight inserted (distance {best_metric:.4f} >= threshold {threshold}).")

    q_scaled = scale_point(q, chosen_weight)
    neighbors = kd_tree_knn_search(chosen_tree, q_scaled, K)
    return neighbors, chosen_weight

# ========================
# 6. Exact kNN 검색 (전체 데이터 대상)
# ========================
def query_exactKNN(q: List[float], w_user: List[float], indexed_data: List[Tuple[List[float], int]], K: int = 3) -> List[tuple]:
    """
    전체 데이터를 w_user로 스케일링한 후, q도 동일 weight로 스케일링하여
    유클리드 거리 기반 정확한 kNN 검색을 수행.
    반환: [(distance, (scaled_feature, index)), ...]
    """
    q_scaled = scale_point(q, w_user)
    dist_list = []
    for (pt, idx) in indexed_data:
        p_scaled = scale_point(pt, w_user)
        d = distance(q_scaled, p_scaled)
        dist_list.append((d, (p_scaled, idx)))
    dist_list.sort(key=lambda x: x[0])
    return dist_list[:K]

# ========================
# 7. kNN 분류 (다수결 투표)
# ========================
def classify_knn(neighbors: List[tuple]) -> str:
    votes = {}
    for d, (pt, idx) in neighbors:
        votes[idx] = votes.get(idx, 0) + 1
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

# ========================
# 9. 저장 및 불러오기
# ========================
def save_forest(weight_list: List[List[float]], kd_tree_list: List[KDNode], filename: str = 'forest.pkl'):
    """현재의 weight list와 kd_tree_list를 파일에 저장"""
    with open(filename, 'wb') as f:
        pickle.dump((weight_list, kd_tree_list), f)
    print("Forest saved to", filename)

def load_forest(filename: str = 'forest.pkl') -> Tuple[Optional[List[List[float]]], Optional[List[KDNode]]]:
    """파일에 저장된 forest가 있으면 불러오고, 없으면 (None, None)을 반환"""
    try:
        with open(filename, 'rb') as f:
            weight_list, kd_tree_list = pickle.load(f)
        print("Forest loaded from", filename)
        return weight_list, kd_tree_list
    except FileNotFoundError:
        print("No saved forest found. A new one will be created.")
        return None, None

# ========================
# 10. 메인: Reader API를 사용하여 Dry Bean 데이터셋 활용
# ========================
if __name__ == "__main__":
    # Reader API를 사용하여 데이터셋 읽기
    data_np, labels_np = rd.read_dataset("dry_bean")
    data = data_np.tolist()
    labels = labels_np.tolist()
    
    n = len(data) # 13611
    dim = len(data[0]) # 16
    index = 7 # 7
    print(f"Dataset loaded with {n} points and {dim} features with {index} classes.")

    # indexed_data: 각 데이터의 (feature, index) 쌍
    indexed_data = [(data[i], labels[i]) for i in range(n)]

    # 10개의 d차원 가중치 벡터를 무작위 생성 (예: 0.05 ~ 0.9, 소수점 2자리)
    weight_list, kd_tree_list = load_forest("forest.pkl")
    if weight_list is None or kd_tree_list is None:
        # 없으면, 10개의 무작위 d차원 weight vector를 생성하고 pre-indexing 수행
        random.seed(42)
        weight_list = [[round(random.uniform(0.05, 0.9), 2) for _ in range(dim)] for _ in range(10)]
        kd_tree_list = build_all_indexes(indexed_data, weight_list)
        save_forest(weight_list, kd_tree_list, "forest.pkl")
    print(f"Weight list : {weight_list}")

    # 사용자 실시간 입력 루프
    while True:
        print(f"\nDataset has {dim} features.")
        print(f"Enter your query vector as {dim} space-separated numbers (e.g. '2.5 3.0 ...'): ")
        q_str = input().strip()
        try:
            q = list(map(float, q_str.split()))
            if len(q) != dim:
                print(f"Please enter exactly {dim} numbers.")
                continue
        except:
            print("Invalid input. Try again.")
            continue

        print(f"Enter your user weight vector as {dim} space-separated numbers (e.g. '0.2 0.4 ...'): ")
        w_str = input().strip()
        try:
            w_user = list(map(float, w_str.split()))
            if len(w_user) != dim:
                print(f"Please enter exactly {dim} numbers.")
                continue
        except:
            print("Invalid input. Try again.")
            continue

        print("Enter the number of neighbors K (e.g. '3'): ")
        k_str = input().strip()
        try:
            K = int(k_str)
        except:
            print("Invalid number. Try again.")
            continue

        # 8-6. AkNN 검색 (Threshold=0.2 적용)
        print("\n=== Approx kNN (AkNN) ===")
        aKNN_results, chosen_weight = query_AkNN(q, w_user, weight_list, kd_tree_list, indexed_data, threshold=0.2, K=K)
        for dist_val, (scaled_pt, idx) in aKNN_results:
            original_pt = invert_scale_point(scaled_pt, chosen_weight)
            print(f"  dist = {dist_val:.4f}, scaled_pt = {scaled_pt}, original_pt = {original_pt}, label = {idx}")
        print(f"  chosen_weight = {chosen_weight}")

        # 8-7. Exact kNN 검색
        if (w_user == chosen_weight):
            print("\nExact kNN search is skipped (same weight as AkNN).")
        else:
            print("\n=== Exact kNN ===")
            exact_results = query_exactKNN(q, w_user, indexed_data, K=K)
            for dist_val, (scaled_pt, idx) in exact_results:
                original_pt = invert_scale_point(scaled_pt, w_user)
                print(f"  dist = {dist_val:.4f}, scaled_pt = {scaled_pt}, original_pt = {original_pt}, label = {idx}")
            print(f"  (user weight = {w_user})")
        
        # 8-8. kNN 분류: 다수결 투표
        pred_label_AkNN = classify_knn(aKNN_results)
        pred_label_exact = classify_knn(exact_results)
        print(f"\nPredicted class (AkNN): {pred_label_AkNN}")
        print(f"Predicted class (Exact kNN): {pred_label_exact}")

        print("Again? (y/n): ")
        if input().strip().lower() != 'y':
            save_forest(weight_list, kd_tree_list, "forest.pkl")
            break
