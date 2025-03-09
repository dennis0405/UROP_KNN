import numpy as np
import sklearn.datasets
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler

def test_read_data():
    # kdd_cup 데이터셋 로드
    data_np, label_np = sklearn.datasets.fetch_kddcup99(download_if_missing=True, return_X_y=True)
    print("Original Data shape:", data_np.shape)
    print("First data sample:", data_np[0])
    print("First label:", label_np[0])
    
    # DataFrame으로 변환 (컬럼명은 임의로 지정)
    df = pd.DataFrame(data_np)
    
    # 범주형 feature 인덱스 (예: 인덱스 1, 2, 3)
    cat_indices = [1, 2, 3]
    
    # 범주형 컬럼은 bytes이므로 문자열로 변환
    for col in cat_indices:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # BinaryEncoder를 사용해 범주형 데이터를 인코딩 (출력 차원을 줄임)
    encoder = BinaryEncoder(cols=cat_indices)
    df_encoded = encoder.fit_transform(df)
    
    # df_encoded에는 범주형 컬럼이 Binary Encoding된 형태로 변환되고, 나머지 컬럼은 그대로 남음
    print("New Data shape after binary encoding:", df_encoded.shape)
    
    # DataFrame을 numpy array로 변환
    new_data = df_encoded.to_numpy().astype(float)
    print('First data sample:', new_data[0])
    
    # 각 피처(열)별 평균과 분산 계산
    means = np.mean(new_data, axis=0)
    variances = np.var(new_data, axis=0)
    
    # 결과 출력: 각 피처 번호와 함께 평균, 분산 표시
    for i in range(new_data.shape[1]):
        print(f"Feature {i+1}: Mean = {means[i]:.3f}, Variance = {variances[i]:.3f}")

def test_read_data_excluding_category():
    # kdd_cup 데이터셋 로드
    data_np, label_np = sklearn.datasets.fetch_kddcup99(download_if_missing=True, return_X_y=True)
    print("Original Data shape:", data_np.shape)
    
    df = pd.DataFrame(data_np)
    
    # 범주형 feature를 제외하고 numeric 데이터만 선택 (drop: 해당 컬럼 제거)
    new_data = df.drop(columns=[1, 2, 3]).astype(float).to_numpy()
    print("New Data shape after excluding category:", new_data.shape)
    
    for i in range(10):
        print(f'data sample {i+1}:', new_data[i])
        print(f'label sample {i+1}:', label_np[i])
    
    # 각 컬럼별 평균과 분산 계산
    means = np.mean(new_data, axis=0)
    variances = np.var(new_data, axis=0)
    
    # 결과 출력: 각 컬럼의 인덱스와 함께 평균, 분산 표시
    for feat in range(new_data.shape[1]):
        print(f"Feature {feat+1}: Mean = {means[feat]:.3f}, Variance = {variances[feat]:.3f}")

    scaler = RobustScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    print("\nNew Data after Robust scaling:")
    means = np.mean(new_data_scaled, axis=0)
    variances = np.var(new_data_scaled, axis=0)
    
    # 결과 출력: 각 컬럼의 인덱스와 함께 평균, 분산 표시
    for feat in range(new_data_scaled.shape[1]):
        print(f"Feature {feat+1}: Mean = {means[feat]:.3f}, Variance = {variances[feat]:.3f}")
    
    print("\nNew Data after Standard scaling:")
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data_scaled)
    means = np.mean(new_data_scaled, axis=0)
    variances = np.var(new_data_scaled, axis=0)
    for feat in range(new_data_scaled.shape[1]):
        print(f"Feature {feat+1}: Mean = {means[feat]:.3f}, Variance = {variances[feat]:.3f}")    
    
    
# 테스트 실행
if __name__ == "__main__":
    #test_read_data()
    test_read_data_excluding_category()
