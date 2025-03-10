import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.datasets import fetch_kddcup99
from kd_tree_forest import save_forest, load_forest
from evaluation import measure_preindex_time, measure_saving_time, measure_loading_time, evaluate_queries

NUM_INITIAL_WEIGHTS = 10
NUM_QUERIES = 100
K = 20
THRESHOLD = 0.2
FOREST_FILENAME = "forest.pkl"

def gui_evaluation():
    import tkinter as tk
    from tkinter import ttk
    import threading
    import matplotlib.pyplot as plt

    root = tk.Tk()
    root.title("AKNN Evaluation")

    # 파라미터 입력 필드 구성
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

    # 실행 결과 출력용 Text 위젯
    output_text = tk.Text(root, width=80, height=20)
    output_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

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
        from time import perf_counter
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
        output_text.insert(tk.END, "Initial weights:\n")
        for i, w in enumerate(weight_list):
            output_text.insert(tk.END, f"Tree {i+1}: weight = {w}\n")
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
        output_text.insert(tk.END, f"Tree loaded: {tree_loaded}, K = {k_val}, threshold = {threshold_val}, initial weights = {len(weight_list)}\n")
        output_text.insert(tk.END, f"Average akNN query time: {avg_q_time:.6f} seconds per query.\n")
        output_text.insert(tk.END, f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.\n")
        output_text.insert(tk.END, f"Classification accuracy (akNN vs exact): {class_acc:.4f}\n")
        output_text.insert(tk.END, f"Average distance error: {avg_q_err:.4f}\n")
        output_text.insert(tk.END, f"Average exact neighbor matches: {avg_exact_matches:.2f} (out of {k_val})\n")
        output_text.insert(tk.END, f"{new_tree_count} new trees were created during the queries.\n")
        output_text.update()

        save_forest(weight_list, kd_tree_list, FOREST_FILENAME)
        output_text.insert(tk.END, "Forest saved.\n")
        output_text.update()

        # 간단한 matplotlib 시각화 (별도 창)
        plt.figure(figsize=(6,4))
        times = [avg_q_time, avg_exact_q_time]
        plt.bar(['akNN Query Time', 'Exact kNN Query Time'], times)
        plt.ylabel("Time (seconds)")
        plt.title("Average Query Times")
        plt.show()

    def on_run_button():
        threading.Thread(target=run_evaluation).start()

    run_button = ttk.Button(root, text="Run Evaluation", command=on_run_button)
    run_button.grid(row=4, column=0, columnspan=2, pady=10)

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
    print(f"Tree loaded: {tree_loaded}, K = {K}, threshold = {THRESHOLD}, initial weights = {len(weight_list)}")
    print(f"Average akNN query time: {avg_q_time:.6f} seconds per query.")
    print(f"Average exact kNN query time: {avg_exact_q_time:.6f} seconds per query.")
    print(f"Classification accuracy (akNN vs exact): {class_acc:.4f}")
    print(f"Average distance error: {avg_q_err:.4f}")
    print(f"Average exact neighbor matches: {avg_exact_matches:.2f} (out of {K})")
    print(f"{new_tree_count} new trees were created during the queries.")

    save_forest(weight_list, kd_tree_list, FOREST_FILENAME)

if __name__ == "__main__":
    # GUI 환경에서 실행하려면 아래 gui_evaluation() 호출 (터미널에서 .py 실행 시 별도 GUI 창이 뜹니다.)
    gui_evaluation()
    # main()  # GUI가 아닌 스크립트 실행 시 main() 호출