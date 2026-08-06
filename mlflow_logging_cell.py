# ============================================================
# MLflow Tracking
# ============================================================
# Paste vào 1 Ô MỚI trong main.ipynb, chạy SAU KHI đã có:
#   1. df_results, num_users  (từ cell đánh giá GD4)
#   2. alpha_val, loyalty_threshold (đã có sẵn từ GD3)
#   3. artifacts/ đã được tạo (đã chạy export_models_from_notebook.py)
#
# Cần cài trước: pip install -r requirements-dashboard.txt

import os

import mlflow

mlflow.set_experiment("ecommerce-recommender")

with mlflow.start_run(run_name="hybrid-waterfall-v1"):
    # --- Hyperparameters ---
    mlflow.log_param("als_factors", 64)
    mlflow.log_param("als_regularization", 0.05)
    mlflow.log_param("als_iterations", 20)
    mlflow.log_param("als_alpha", alpha_val)
    mlflow.log_param("hybrid_w_als", 0.6)
    mlflow.log_param("hybrid_w_content", 0.4)
    mlflow.log_param("loyalty_threshold", loyalty_threshold)
    mlflow.log_metric("evaluated_users", num_users)

    # --- Metrics: Precision@10 / Recall@10 cho từng phương pháp ---
    for _, row in df_results.iterrows():
        slug = row["Phương pháp"].split(" ")[0].lower().strip("()")
        mlflow.log_metric(f"{slug}_precision_at_10", row["Precision"])
        mlflow.log_metric(f"{slug}_recall_at_10", row["Recall"])

    # --- Lưu bảng kết quả để Streamlit dashboard đọc lại được ---
    df_results.to_csv("evaluation_results.csv", index=False)
    mlflow.log_artifact("evaluation_results.csv")

    # --- Log toàn bộ model artifacts (.pkl) nếu đã export ---
    if os.path.isdir("artifacts"):
        mlflow.log_artifacts("artifacts", artifact_path="model_artifacts")

    print("✅ Đã log xong vào MLflow!")
    print("Gõ 'mlflow ui' trong terminal, rồi mở http://localhost:5000 để xem.")
