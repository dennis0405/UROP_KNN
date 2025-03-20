from typing import List
import numpy as np

# KNN result classification (weighted)
def classify_knn_weighted(neighbors: List[tuple], labels: np.ndarray) -> str:
    epsilon = 1e-6
    
    distances = np.array([dist for (dist, _) in neighbors])
    indices = np.array([idx for (_, idx) in neighbors])
    
    # weight 계산: 1/(distance + epsilon)
    weights = 1.0 / (distances + epsilon)
    
    # 각 neighbor의 label을 가져옴
    neighbor_labels = labels[indices]
    
    # unique한 label별로 가중치 합산
    unique_labels, inverse = np.unique(neighbor_labels, return_inverse=True)
    vote_sums = np.bincount(inverse, weights=weights)
    
    # 가중치 합산이 최대인 label 선택
    best_label = unique_labels[np.argmax(vote_sums)]
    return best_label