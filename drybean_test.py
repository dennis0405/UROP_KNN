import reader
import numpy as np

def test_read_data():
    data_np, label_np = reader.read_dataset("dry_bean")
    print("Data shape:", data_np.shape)
    print("First data sample:", data_np[0])
    print("First label:", label_np[0])
    
    # 각 피처(열)별 평균과 분산 계산 (axis=0)
    means = np.mean(data_np, axis=0)
    variances = np.var(data_np, axis=0)
    
    # 결과 출력: 각 피처 번호와 함께 평균, 분산 표시
    for i in range(data_np.shape[1]):
        print(f"Feature {i+1}: Mean = {means[i]:.3f}, Variance = {variances[i]:.3f}")

# 테스트 실행
if __name__ == "__main__":
    test_read_data()

    
    