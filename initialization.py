import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import RobustScaler, StandardScaler
from scaling import distance_in_weight_space

# 초기 weight list 생성
def generate_weight_list(num_trees: int, dim: int) -> np.ndarray:
    threshold = 0.2
    weights = []
    new_weight = np.round(np.random.uniform(0, 1, size=dim), 3)
    weights.append(new_weight)
    
    while len(weights) < num_trees:
        new_weight = np.round(np.random.uniform(0, 1, size=dim), 3)
        _, best_metric = distance_in_weight_space(new_weight, np.array(weights))
        
        if best_metric >= threshold:
            weights.append(new_weight)
        
    return np.array(weights)

# kdd_cup dataset 가져오고, preprocessing
def fetch_data():
    data, labels = fetch_kddcup99(download_if_missing=True, return_X_y=True)
    df = pd.DataFrame(data)
    
    df = df.drop(columns=[1, 2, 3])
    data = df.astype(float).to_numpy()
        
    scaler = RobustScaler()
    data = scaler.fit_transform(data)
        
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
        
    decode = np.vectorize(lambda x: x.decode('utf-8'))
    labels = decode(labels)
    
    return data, labels

def generate_tree_usage_list(num_trees: int) -> list[int]:
    return ([0] * num_trees)