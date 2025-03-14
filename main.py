import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.datasets import fetch_kddcup99
from kd_tree_forest import save_forest, load_forest
from evaluation import measure_preindex_time, measure_saving_time, measure_loading_time, evaluate_queries
from kd_tree_forest import check_forest
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
import copy
import os
import csv

NUM_INITIAL_WEIGHTS = 20
NUM_QUERIES = 300
K = 20
THRESHOLD = 0.15
FOREST_FILENAME = "forest.pkl"

def gui_evaluation():
    root = tk.Tk()
    root.title("AKNN Evaluation")

    tk.Label(root, text="Initial Weights:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    initial_weights_var = tk.StringVar(value=str(NUM_INITIAL_WEIGHTS))
    tk.Entry(root, textvariable=initial_weights_var).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Number of Queries:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    num_queries_var = tk.StringVar(value=str(NUM_QUERIES))
    tk.Entry(root, textvariable=num_queries_var).grid(row=1, column=1, padx=5, pady=5)

    tk.Label(root, text="K:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    k_var = tk.StringVar(value=str(K))
    tk.Entry(root, textvariable=k_var).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="Threshold:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    threshold_var = tk.StringVar(value=str(THRESHOLD))
    tk.Entry(root, textvariable=threshold_var).grid(row=3, column=1, padx=5, pady=5)

    # 추가: Threshold Experiment를 위한 범위 입력 필드
    tk.Label(root, text="Threshold Start:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
    thresh_start_var = tk.StringVar(value="0.0")
    tk.Entry(root, textvariable=thresh_start_var).grid(row=4, column=1, padx=5, pady=5)

    tk.Label(root, text="Threshold End:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
    thresh_end_var = tk.StringVar(value="1.0")
    tk.Entry(root, textvariable=thresh_end_var).grid(row=5, column=1, padx=5, pady=5)

    tk.Label(root, text="Threshold Step:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
    thresh_step_var = tk.StringVar(value="0.1")
    tk.Entry(root, textvariable=thresh_step_var).grid(row=6, column=1, padx=5, pady=5)

    # 실행 결과 출력용 Text 위젯
    output_text = tk.Text(root, width=80, height=20)
    output_text.grid(row=9, column=0, columnspan=2, padx=5, pady=5)
    
    # 평가 실행 함수 (별도 스레드에서 실행)
    def run_evaluation():
        try:
            initial_weights = int(initial_weights_var.get())
            num_queries_val = int(num_queries_var.get())
            k_val = int(k_var.get())
            threshold_val = float(threshold_var.get())
        except Exception as e:
            output_text.insert(tk.END, "Error in input values: {}\n".format(e))
            return

        output_text.insert(tk.END, "Running evaluation with parameters:\n")
        output_text.insert(tk.END, f"Initial Weights: {initial_weights}\n")
        output_text.insert(tk.END, f"Number of Queries: {num_queries_val}\n")
        output_text.insert(tk.END, f"K: {k_val}\n")
        output_text.insert(tk.END, f"Threshold: {threshold_val}\n")
        output_text.update()

        # 데이터 로드 및 전처리
        output_text.insert(tk.END, "Loading and preprocessing data...\n")
        output_text.update()
        data, labels = fetch_kddcup99(download_if_missing=True, return_X_y=True)
        output_text.insert(tk.END, f"Original kdd_cup Data shape: {data.shape}\n")
        output_text.update()

        df = pd.DataFrame(data)
        df = df.drop(columns=[1, 2, 3])
        data_proc = df.astype(float).to_numpy()

        scaler = RobustScaler()
        data_proc = scaler.fit_transform(data_proc)
        scaler = StandardScaler()
        data_proc = scaler.fit_transform(data_proc)

        decode = np.vectorize(lambda x: x.decode('utf-8'))
        labels_proc = decode(labels).tolist()

        n, dim = data_proc.shape
        output_text.insert(tk.END, f"Dataset loaded with {n} points and {dim} features (numeric only, standardized).\n")
        output_text.update()

        # Forest 로드 또는 생성
        weight_list, kd_tree_list = load_forest(FOREST_FILENAME)
        tree_loaded = True
        if weight_list is None or kd_tree_list is None:
            tree_loaded = False
            weight_list = []
            for _ in range(initial_weights):
                weight = np.round(np.random.uniform(0, 1, size=dim), 3).tolist()
                weight_list.append(weight)
        
        output_text.insert(tk.END, "Initialized weights and trees:\n")
        output_text.insert(tk.END, "Total {} trees\n".format(len(weight_list)))
        output_text.update()

        preindex_time, kd_tree_list = measure_preindex_time(data_proc, weight_list)
        output_text.insert(tk.END, f"Pre-indexing time for forest: {preindex_time:.4f} seconds.\n")
        saving_time = measure_saving_time(weight_list, kd_tree_list, FOREST_FILENAME)
        output_text.insert(tk.END, f"Saving time for forest: {saving_time:.4f} seconds.\n")
        loading_time = measure_loading_time(FOREST_FILENAME)
        output_text.insert(tk.END, f"Loading time for forest: {loading_time:.4f} seconds.\n")
        output_text.update()

        avg_q_time, avg_exact_q_time, class_acc, avg_q_err, avg_exact_matches, new_tree_count = evaluate_queries(
            num_queries_val, labels_proc, data_proc, weight_list, kd_tree_list, k_val, threshold_val
        )

        output_text.insert(tk.END, "\n=== Query Evaluation over {} queries ===\n".format(num_queries_val))
        output_text.insert(tk.END, f"Tree loaded: {tree_loaded}, K = {k_val}, threshold = {threshold_val}, initial weights = {initial_weights_var}\n")
        output_text.insert(tk.END, f"Average akNN query time: {avg_q_time:.6f} seconds per query.\n")
        output_text.insert(tk.END, f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.\n")
        output_text.insert(tk.END, f"Classification accuracy (akNN vs exact): {class_acc:.4f}\n")
        output_text.insert(tk.END, f"Average distance error: {avg_q_err:.4f}\n")
        output_text.insert(tk.END, f"Average exact neighbor matches: {avg_exact_matches:.2f} (out of {k_val})\n")
        output_text.insert(tk.END, f"{new_tree_count} new trees were created during the queries.\n")
        output_text.update()

        # forest를 save하는 데 저장 용량이 너무 크기 때문에, 실시간 tracking은 일단 보류
        #save_forest(weight_list, kd_tree_list, FOREST_FILENAME)
        #output_text.insert(tk.END, "Forest saved.\n")
        #output_text.update()

        # 간단한 matplotlib 시각화 (별도 창), 추가 필요
        plt.figure(figsize=(6,4))
        times = [avg_q_time, avg_exact_q_time]
        plt.bar(['akNN Query Time', 'Exact kNN Query Time'], times)
        plt.ylabel("Time (seconds)")
        plt.title("Average Query Times")
        plt.show()

    def run_threshold_evaluation():
        try:
            initial_weights = int(initial_weights_var.get())
            num_queries_val = int(num_queries_var.get())
            k_val = int(k_var.get())
            thresh_start = float(thresh_start_var.get())
            thresh_end = float(thresh_end_var.get())
            thresh_step = float(thresh_step_var.get())
        except Exception as e:
            output_text.insert(tk.END, "Error in threshold experiment inputs: {}\n".format(e))
            return

        output_text.insert(tk.END, f"Running threshold experiment from {thresh_start} to {thresh_end} (step {thresh_step})\n")
        output_text.update()

        # 데이터 로드 및 전처리
        data, labels = fetch_kddcup99(download_if_missing=True, return_X_y=True)
        df = pd.DataFrame(data)
        df = df.drop(columns=[1, 2, 3])
        data_proc = df.astype(float).to_numpy()
        scaler = RobustScaler()
        data_proc = scaler.fit_transform(data_proc)
        scaler = StandardScaler()
        data_proc = scaler.fit_transform(data_proc)
        decode = np.vectorize(lambda x: x.decode('utf-8'))
        labels_proc = decode(labels).tolist()
        n, dim = data_proc.shape

        # 결과를 저장할 리스트들
        thresh_values = []
        aknn_times = []
        exact_times = []
        class_accs = []
        query_errors = []
        exact_matches_list = []
        new_tree_counts = []
        
        weight_list, kd_tree_list = load_forest(FOREST_FILENAME)
        if weight_list is None or kd_tree_list is None:
            weight_list = []
            for _ in range(initial_weights):
                weight = np.round(np.random.uniform(0, 1, size=dim), 3).tolist()
                weight_list.append(weight)
            preindex_time, kd_tree_list = measure_preindex_time(data_proc, weight_list)
            save_forest(weight_list, kd_tree_list, FOREST_FILENAME)
        else:
            preindex_time, _ = measure_preindex_time(data_proc, weight_list)
        
        output_text.insert(tk.END, "Initialized weights and trees:\n")
        output_text.insert(tk.END, "Total {} trees\n".format(len(weight_list)))
        output_text.insert(tk.END, f"Pre-indexing time for forest: {preindex_time:.4f} seconds.\n\n")
        output_text.update()
        
        thresh = thresh_start
        while thresh <= thresh_end + 1e-8:
            output_text.insert(tk.END, f"Evaluating threshold = {thresh:.3f}\n")
            output_text.update()
            
            for _ in range(3):
                avg_q_time, avg_exact_q_time, class_acc, avg_q_err, avg_exact_matches, new_tree_count = evaluate_queries(
                    num_queries_val, labels_proc, data_proc, copy.deepcopy(weight_list), copy.deepcopy(kd_tree_list), k_val, thresh
                )
                thresh_values.append(thresh)
                aknn_times.append(avg_q_time)
                exact_times.append(avg_exact_q_time)
                class_accs.append(class_acc)
                query_errors.append(avg_q_err)
                exact_matches_list.append(avg_exact_matches)
                new_tree_counts.append(new_tree_count)
                
            thresh += thresh_step

        # 결과 폴더 생성
        folder_name = f"test_Q{num_queries_val}_W{initial_weights}_K{k_val}_Th{thresh_start}_{thresh_end}_{thresh_step}"
        base_folder = "test_result"
        full_folder = os.path.join(base_folder, folder_name)
        os.makedirs(full_folder, exist_ok=True)

        csv_path = os.path.join(full_folder, "results.csv")
        with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # header 작성
            csv_writer.writerow(["Threshold", "akNN_Time", "Exact_Time", "Class_Acc", "Query_Error", "Exact_Matches", "New_Tree_Count"])
            for i in range(len(thresh_values)):
                csv_writer.writerow([thresh_values[i], aknn_times[i], exact_times[i],
                                    class_accs[i], query_errors[i], exact_matches_list[i], new_tree_counts[i]])
        
        """
        # ─────────────────────────────────────────────────────────────
        # 그래프 1) threshold vs akNN & exact query time
        # ─────────────────────────────────────────────────────────────
        plt.figure(figsize=(8,6))
        plt.plot(thresh_values, aknn_times, marker='o', linestyle='-', color='blue', label='akNN Query Time')
        plt.plot(thresh_values, exact_times, marker='s', linestyle='--', color='red', label='Exact kNN Query Time')
        plt.xlabel("Threshold")
        plt.ylabel("Query Time (seconds)")
        plt.title("Query Time vs Threshold")
        plt.legend()
        plt.grid(True)
        graph1_path = os.path.join(full_folder, "query_times.png")
        plt.savefig(graph1_path)
        plt.close()

        # ─────────────────────────────────────────────────────────────
        # 그래프 2) threshold vs classification accuracy, avg query error, exact matches
        # ─────────────────────────────────────────────────────────────
        plt.figure(figsize=(8,6))
        plt.plot(thresh_values, class_accs, marker='^', linestyle='-', color='green', label='Classification Accuracy')
        plt.plot(thresh_values, query_errors, marker='d', linestyle='--', color='orange', label='Avg Query Error')
        plt.plot(thresh_values, exact_matches_list, marker='x', linestyle='-.', color='purple', label='Exact Neighbor Matches')
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Performance Metrics vs Threshold")
        plt.legend()
        plt.grid(True)
        graph2_path = os.path.join(full_folder, "performance_metrics.png")
        plt.savefig(graph2_path)
        plt.close()

        # ─────────────────────────────────────────────────────────────
        # 그래프 3) threshold vs new tree count
        # ─────────────────────────────────────────────────────────────
        plt.figure(figsize=(8,6))
        plt.plot(thresh_values, new_tree_counts, marker='*', linestyle='-', color='magenta', label='New Tree Count')
        plt.xlabel("Threshold")
        plt.ylabel("New Trees Created")
        plt.title("New Trees Created vs Threshold")
        plt.legend()
        plt.grid(True)
        graph3_path = os.path.join(full_folder, "tree_creation.png")
        plt.savefig(graph3_path)
        plt.close()
        """

        output_text.insert(tk.END, f"Threshold experiment completed.\nSaved in folder: {full_folder}\n")
        output_text.update()
    
    def on_run_button():
        threading.Thread(target=run_evaluation).start()

    def on_run_threshold_button():
        threading.Thread(target=run_threshold_evaluation).start()
        
    run_button = ttk.Button(root, text="Run Evaluation", command=on_run_button)
    run_button.grid(row=7, column=0, columnspan=2, pady=10)
    
    run_thresh_button = ttk.Button(root, text="Run Threshold Experiment", command=on_run_threshold_button)
    run_thresh_button.grid(row=8, column=0, columnspan=2, pady=10)

    root.mainloop()

# Main function (GUI가 아닌 환경에서 실행)
def main() -> None:
        
    # 데이터 로드 및 전처리
    data, labels = fetch_kddcup99(download_if_missing=True, return_X_y=True)
    print(f"Original kdd_cup Data shape: {data.shape}")

    df = pd.DataFrame(data)
    df = df.drop(columns=[1, 2, 3])
    data_proc = df.astype(float).to_numpy()

    scaler = RobustScaler()
    data_proc = scaler.fit_transform(data_proc)
    scaler = StandardScaler()
    data_proc = scaler.fit_transform(data_proc)

    decode = np.vectorize(lambda x: x.decode('utf-8'))
    labels_proc = decode(labels).tolist()

    n, dim = data_proc.shape
    print(f"Dataset loaded with {n} points and {dim} features (numeric only, standardized).")

    # Forest 로드 혹은 생성
    weight_list, kd_tree_list = load_forest(FOREST_FILENAME)
    tree_loaded = True
    if weight_list is None or kd_tree_list is None:
        tree_loaded = False
        weight_list = []
        for _ in range(NUM_INITIAL_WEIGHTS):
            weight = np.round(np.random.uniform(0, 1, size=dim), 3).tolist()
            weight_list.append(weight)

    for i, w in enumerate(weight_list):
        print(f"Tree {i+1}: weight = {w}")

    preindex_time, kd_tree_list = measure_preindex_time(data_proc, weight_list)
    print(f"Pre-indexing time for forest: {preindex_time:.4f} seconds.")

    saving_time = measure_saving_time(weight_list, kd_tree_list, FOREST_FILENAME)
    print(f"Saving time for forest: {saving_time:.4f} seconds.")

    loading_time = measure_loading_time(FOREST_FILENAME)
    print(f"Loading time for forest: {loading_time:.4f} seconds.")

    avg_q_time, avg_exact_q_time, class_acc, avg_q_err, avg_exact_matches, new_tree_count = evaluate_queries(
        NUM_QUERIES, labels_proc, data_proc, weight_list, kd_tree_list, K, THRESHOLD
    )

    print("\n=== Query Evaluation over {} queries ===".format(NUM_QUERIES))
    print(f"Tree loaded: {tree_loaded}, K = {K}, threshold = {THRESHOLD}, initial weights = {NUM_INITIAL_WEIGHTS}")
    print(f"Average akNN query time: {avg_q_time:.6f} seconds per query.")
    print(f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.")
    print(f"Classification accuracy (akNN vs exact): {class_acc:.4f}")
    print(f"Average distance error: {avg_q_err:.4f}")
    print(f"Average exact neighbor matches: {avg_exact_matches:.2f})")
    print(f"{new_tree_count} new trees were created during the queries.")

    # forest tracking은 일단 보류류
    #save_forest(weight_list, kd_tree_list, FOREST_FILENAME)

if __name__ == "__main__":
    # GUI 환경에서 실행하려면 아래 gui_evaluation() 호출
    gui_evaluation()
    
    #check_forest()
    
    # GUI가 아닌 스크립트 실행 시 main() 호출
    #main()