from typing import List

# KNN result classification (weighted)
def classify_knn_weighted(neighbors: List[tuple], labels: List[str]) -> str:
    epsilon = 1e-6
    vote_dict = {}
    for dist, neighbor in neighbors:
        idx = neighbor[1]  
        label = labels[idx]
        weight = 1.0 / (dist + epsilon)
        vote_dict[label] = vote_dict.get(label, 0) + weight
    return max(vote_dict.items(), key=lambda x: x[1])[0]