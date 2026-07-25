"""
Load các artifact đã export từ notebook (xem export_models_from_notebook.py)
và tái hiện đúng logic Hybrid Waterfall mô tả trong README hiện tại:
  - loyal (>=2 tương tác, có trong ALS)  -> Weighted Hybrid (ALS 60% + Content 40%)
  - casual (< 2 tương tác)               -> Content-Based trên sản phẩm xem gần nhất
  - cold-start (chưa có lịch sử)         -> Trending
"""
import pickle
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

ALS_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4
TOP_N = 10


def _load(name: str):
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {name}. Chạy export_models_from_notebook.py "
            f"trong main.ipynb trước, rồi copy thư mục artifacts/ vào đây."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


class RecommenderService:
    def __init__(self):
        self.als_model = _load("als_model.pkl")
        self.user_item_matrix = _load("user_item_matrix.pkl")
        self.user_id_to_idx = _load("user_id_to_idx.pkl")
        self.idx_to_product_id = _load("idx_to_product_id.pkl")

        self.cosine_sim = _load("cosine_sim.pkl")
        self.product_id_to_idx = _load("product_id_to_idx.pkl")
        self.idx_to_product_id_content = {v: k for k, v in self.product_id_to_idx.items()}

        self.trending_top10 = _load("trending_top10.pkl")
        self.loyal_user_ids = _load("loyal_user_ids.pkl")

        # TODO: thay bằng query PostgreSQL thật tới view user_last_viewed_product
        # (feature_engineering.sql) — tạm thời để dict rỗng, cần nối DB ở đây.
        self.user_last_viewed = {}  # {user_id: product_id}

    # ---------- Model A: ALS ----------
    def _recommend_als(self, user_id: int, n: int = TOP_N):
        idx = self.user_id_to_idx.get(user_id)
        if idx is None:
            return []
        product_ids, scores = self.als_model.recommend(
            idx, self.user_item_matrix[idx], N=n
        )
        return [
            (self.idx_to_product_id[pid], float(s))
            for pid, s in zip(product_ids, scores)
        ]

    # ---------- Model B: Content-Based ----------
    def _recommend_content(self, product_id: int, n: int = TOP_N):
        idx = self.product_id_to_idx.get(product_id)
        if idx is None:
            return []
        sims = self.cosine_sim[idx]
        top_idx = np.argsort(sims)[::-1][1: n + 1]  # bỏ chính nó (index 0)
        return [
            (self.idx_to_product_id_content[i], float(sims[i]))
            for i in top_idx
        ]

    # ---------- Baseline: Trending ----------
    def _recommend_trending(self, n: int = TOP_N):
        return [(pid, None) for pid in self.trending_top10[:n]]

    # ---------- Hybrid waterfall (logic chính) ----------
    def recommend(self, user_id: int, n: int = TOP_N):
        # Nhóm 1: loyal -> weighted hybrid
        if user_id in self.loyal_user_ids:
            als_results = dict(self._recommend_als(user_id, n=n * 2))
            last_viewed = self.user_last_viewed.get(user_id)
            content_results = dict(
                self._recommend_content(last_viewed, n=n * 2) if last_viewed else []
            )
            all_products = set(als_results) | set(content_results)
            scored = [
                (
                    pid,
                    ALS_WEIGHT * als_results.get(pid, 0)
                    + CONTENT_WEIGHT * content_results.get(pid, 0),
                )
                for pid in all_products
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            source = "hybrid"
            results = scored[:n]

        # Nhóm 2: casual -> content-based trên sản phẩm xem gần nhất
        elif user_id in self.user_last_viewed:
            results = self._recommend_content(self.user_last_viewed[user_id], n=n)
            source = "content_based"

        # Nhóm 3: cold-start -> trending
        else:
            results = self._recommend_trending(n=n)
            source = "trending"

        return {
            "user_id": user_id,
            "source": source,
            "recommendations": [
                {"product_id": pid, "score": score} for pid, score in results
            ],
        }
