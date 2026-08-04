# ============================================================
# ĐÂY LÀ CHỖ CHÍNH KHANH CẦN THÊM CODE
# ============================================================
# Copy toàn bộ nội dung file này vào Ô CUỐI CÙNG của main.ipynb,
# NGAY SAU khi đã train xong GD3 (Baseline, ALS, TF-IDF, Hybrid).
# Chạy 1 lần để xuất model ra file .pkl cho FastAPI load.
#
# THAY TÊN BIẾN bên dưới (model_a, cosine_sim, v.v.) đúng theo
# tên biến thật trong notebook của Khanh — mình đặt tên theo
# đúng mô tả trong README_RecommendationSystem.md, nhưng notebook
# thật có thể đặt tên khác, cần đối chiếu lại.

import pickle
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- 1. Model A: ALS (thư viện `implicit`) ---
# Biến cần có sẵn trong notebook: model đã train + ma trận user-item
# + 2 dict map giữa user_id/product_id thật và index nội bộ của ALS
with open(f"{ARTIFACT_DIR}/als_model.pkl", "wb") as f:
    pickle.dump(model_als, f)                      # <-- THAY bằng biến ALS model thật của Khanh

with open(f"{ARTIFACT_DIR}/user_item_matrix.pkl", "wb") as f:
    pickle.dump(matrix_user_item, f)                # <-- ma trận sparse user-item dùng để train ALS

with open(f"{ARTIFACT_DIR}/user_id_to_idx.pkl", "wb") as f:
    pickle.dump(user_to_idx, f)                # <-- dict {user_id thật: index trong ma trận}

with open(f"{ARTIFACT_DIR}/idx_to_product_id.pkl", "wb") as f:
    pickle.dump(idx_to_item, f)             # <-- dict {index trong ma trận: product_id thật}

# --- 2. Model B: Content-Based (TF-IDF + Cosine Similarity) ---
with open(f"{ARTIFACT_DIR}/cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)             # <-- ma trận 5844x5844 cosine similarity

with open(f"{ARTIFACT_DIR}/product_id_to_idx.pkl", "wb") as f:
    pickle.dump(indices, f)             # <-- dict {product_id: index trong ma trận content}

# --- 3. Baseline: Trending ---
with open(f"{ARTIFACT_DIR}/trending_top10.pkl", "wb") as f:
    pickle.dump(top_10_trending, f)           # <-- list product_id top-10 phổ biến nhất

# --- 4. Danh sách user thuộc nhóm "loyal" (đã có trong ALS) ---
# Dùng để quyết định user nào chạy Weighted Hybrid vs Content-Based fallback
with open(f"{ARTIFACT_DIR}/loyal_user_ids.pkl", "wb") as f:
    pickle.dump(set(loyal_ids), f)           # <-- set user_id nằm trong nhóm >=2 tương tác + có trong ALS

print("Đã export xong toàn bộ model vào thư mục artifacts/. Copy thư mục này")
print("vào cùng cấp với app.py rồi chạy: uvicorn app:app --reload")
