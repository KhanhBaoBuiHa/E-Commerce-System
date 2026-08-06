"""
Dashboard demo cho Hệ thống Gợi ý Sản phẩm.

Chạy:
    streamlit run streamlit_app.py

Yêu cầu trước khi chạy:
    - đã có artifacts/ (xem export_models_from_notebook.py)
    - .env đã cấu hình DATABASE_URL đúng (xem README)
    - (tuỳ chọn) đã chạy mlflow_logging_cell.py trong notebook để có
      file evaluation_results.csv -> tab "So sánh mô hình" sẽ dùng số liệu
      thật thay vì số liệu tĩnh fallback.
"""
import pandas as pd
import streamlit as st

from model_utils import RecommenderService

st.set_page_config(page_title="Recommender Dashboard", page_icon="🛒", layout="wide")

SOURCE_LABELS = {
    "hybrid": ("🟢 Khách thân thiết", "Weighted Hybrid (ALS 60% + Content 40%)"),
    "content_based": ("🟡 Khách vãng lai", "Content-Based (dựa trên sản phẩm xem gần nhất)"),
    "trending": ("🔵 Khách mới (Cold Start)", "Trending (sản phẩm bán chạy nhất)"),
}

FALLBACK_EVAL = pd.DataFrame(
    {
        "Phương pháp": [
            "Hybrid (Kết hợp)",
            "Content-Based",
            "ALS (Collaborative)",
            "Trending (Baseline)",
        ],
        "Precision": [0.0476, 0.0408, 0.0190, 0.0143],
        "Recall": [0.3333, 0.2667, 0.1333, 0.1000],
    }
)


@st.cache_resource
def load_recommender():
    return RecommenderService()


st.title("🛒 Hệ Thống Gợi Ý Sản Phẩm — Dashboard Demo")

tab_demo, tab_eval = st.tabs(["🔍 Thử gợi ý", "📊 So sánh mô hình"])

with tab_demo:
    st.subheader("Nhập User ID để xem gợi ý")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input(
            "User ID", placeholder="Ví dụ: 1515915625353234047"
        )
    with col2:
        top_n = st.slider("Số lượng gợi ý", min_value=3, max_value=20, value=10)

    if st.button("Gợi ý ngay", type="primary"):
        if not user_id_input.strip():
            st.warning("Nhập User ID trước đã nha.")
        else:
            try:
                recommender = load_recommender()
                user_id = int(user_id_input.strip())
                result = recommender.recommend(user_id=user_id, n=top_n)

                label, desc = SOURCE_LABELS.get(
                    result["source"], (result["source"], "")
                )
                st.info(f"**Nhóm khách hàng:** {label}\n\n**Chiến lược áp dụng:** {desc}")

                recs_df = pd.DataFrame(result["recommendations"])
                st.dataframe(recs_df, use_container_width=True)
            except FileNotFoundError as e:
                st.error(f"Chưa có model: {e}")
            except Exception as e:
                st.error(f"Lỗi: {e}")

with tab_eval:
    st.subheader("Precision@10 / Recall@10 — 4 phương pháp")

    try:
        eval_df = pd.read_csv("evaluation_results.csv")
        st.caption("Nguồn: evaluation_results.csv (log từ MLflow)")
    except FileNotFoundError:
        eval_df = FALLBACK_EVAL
        st.caption(
            "Chưa có evaluation_results.csv — đang dùng số liệu tĩnh từ README. "
            "Chạy mlflow_logging_cell.py trong notebook để có số liệu thật, cập nhật tự động."
        )

    st.dataframe(eval_df, use_container_width=True)
    st.bar_chart(eval_df.set_index("Phương pháp")[["Precision", "Recall"]])
