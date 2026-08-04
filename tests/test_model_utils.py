"""
Unit tests cho logic Hybrid Waterfall trong model_utils.py.

Không cần artifacts/ thật hay PostgreSQL thật để chạy — mọi thứ được
"giả lập" (fake) bằng monkeypatch, nên test chạy nhanh và chạy được
trên CI (GitHub Actions) mà không cần setup gì thêm.
"""
import numpy as np
import pytest

import model_utils


class FakeALS:
    """Giả lập model ALS đã train — luôn 'gợi ý' product index 0 và 1."""

    def recommend(self, idx, vector, N=10):
        ids = [0, 1][:N]
        scores = [0.9, 0.5][: len(ids)]
        return ids, scores


class FakeMatrix:
    """Giả lập ma trận user-item — chỉ cần hỗ trợ matrix[idx]."""

    def __getitem__(self, idx):
        return None


# product_id 100 <-> idx 0, product_id 200 <-> idx 1, product_id 300 <-> idx 2
FAKE_ARTIFACTS = {
    "als_model.pkl": FakeALS(),
    "user_item_matrix.pkl": FakeMatrix(),
    "user_id_to_idx.pkl": {1: 0, 2: 1},  # user 1, 2 có trong ALS
    "idx_to_product_id.pkl": {0: 100, 1: 200},
    "cosine_sim.pkl": np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    ),
    "product_id_to_idx.pkl": {100: 0, 200: 1, 300: 2},
    "trending_top10.pkl": [900, 901, 902],
    "loyal_user_ids.pkl": {1},  # chỉ user 1 là "loyal"
}

# user 2 = casual, đã xem product 200 gần nhất
# user 3 = không loyal, không có trong last_viewed -> sẽ rơi vào cold-start
FAKE_LAST_VIEWED = {2: 200}


@pytest.fixture
def recommender(monkeypatch):
    """Trả về 1 RecommenderService với toàn bộ artifact + DB bị giả lập."""
    monkeypatch.setattr(model_utils, "_load", lambda name: FAKE_ARTIFACTS[name])
    monkeypatch.setattr(model_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(model_utils, "create_engine", lambda url: None)

    class FakeDF:
        user_id = list(FAKE_LAST_VIEWED.keys())
        product_id = list(FAKE_LAST_VIEWED.values())

    monkeypatch.setattr(model_utils.pd, "read_sql", lambda query, engine: FakeDF())

    return model_utils.RecommenderService()


# ---------- Test logic phân nhóm Hybrid Waterfall ----------

def test_loyal_user_gets_hybrid_recommendation(recommender):
    result = recommender.recommend(user_id=1, n=5)
    assert result["source"] == "hybrid"
    assert result["user_id"] == 1
    assert len(result["recommendations"]) > 0


def test_casual_user_gets_content_based_recommendation(recommender):
    result = recommender.recommend(user_id=2, n=5)
    assert result["source"] == "content_based"


def test_cold_start_user_gets_trending_recommendation(recommender):
    result = recommender.recommend(user_id=999999, n=3)
    assert result["source"] == "trending"
    returned_ids = [r["product_id"] for r in result["recommendations"]]
    assert returned_ids == [900, 901, 902]


def test_trending_respects_requested_top_n(recommender):
    result = recommender.recommend(user_id=999999, n=2)
    assert len(result["recommendations"]) == 2


# ---------- Test từng "mảnh" logic riêng lẻ ----------

def test_content_recommendation_excludes_the_product_itself(recommender):
    recs = recommender._recommend_content(product_id=200, n=2)
    returned_ids = [pid for pid, _ in recs]
    assert 200 not in returned_ids


def test_content_recommendation_unknown_product_returns_empty(recommender):
    recs = recommender._recommend_content(product_id=999, n=5)
    assert recs == []


def test_als_recommendation_unknown_user_returns_empty(recommender):
    recs = recommender._recommend_als(user_id=999999, n=5)
    assert recs == []
