import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV 파일 경로 설정
folder_name = "test_Q300_W20_K20_Th0.135_0.145_0.001"
csv_path = os.path.join("test_result", folder_name, "results.csv")

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 각 threshold별로 median 값 계산 (groupby를 이용)
df_median = df.groupby("Threshold", as_index=False).median()

# 결과 확인
print(df_median.head())

# 각 metric에 대한 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(df_median["Threshold"], df_median["akNN_Time"], marker='o', color='blue', linestyle='-', label='akNN Query Time')
plt.plot(df_median["Threshold"], df_median["Exact_Time"], marker='s', color='red', linestyle='--', label='Exact kNN Query Time')
plt.xlabel("Threshold")
plt.ylabel("Query Time (seconds)")
plt.title("Query Time vs Threshold (Median)")
plt.legend()
plt.grid(True)
graph_path = os.path.join(os.path.dirname(csv_path), "query_time.png")
plt.savefig(graph_path)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df_median["Threshold"], df_median["Class_Acc"], marker='^', linestyle='-', label='Classification Accuracy')
plt.plot(df_median["Threshold"], df_median["Query_Error"], marker='d', linestyle='--', label='Average Query Error')
plt.plot(df_median["Threshold"], df_median["Exact_Matches"], marker='x', linestyle='-.', label='Exact Neighbor Matches')
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Performance Metrics vs Threshold (Median)")
plt.legend()
plt.grid(True)
graph_path = os.path.join(os.path.dirname(csv_path), "performance_metrics.png")
plt.savefig(graph_path)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df_median["Threshold"], df_median["New_Tree_Count"], marker='*', linestyle='-', label='New Tree Count')
plt.xlabel("Threshold")
plt.ylabel("New Trees Created")
plt.title("New Tree Count vs Threshold (Median)")
plt.legend()
plt.grid(True)
graph_path = os.path.join(os.path.dirname(csv_path), "tree_creation.png")
plt.savefig(graph_path)
plt.close()
