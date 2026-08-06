# Hệ Thống Gợi Ý Sản Phẩm 🛒

[![CI](https://github.com/KhanhBaoBuiHa/E-Commerce-System/actions/workflows/ci.yml/badge.svg)](https://github.com/KhanhBaoBuiHa/E-Commerce-System/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)

Đồ án môn **Python cho Khoa học Dữ liệu** — Lớp 23TTH, Khoa Toán - Tin học
Giảng viên: ThS. Hà Văn Thảo

---

## Giới thiệu

Trong bối cảnh quá tải thông tin của thương mại điện tử, hệ thống gợi ý đóng vai trò cá nhân hóa trải nghiệm mua sắm và tối ưu tỷ lệ chuyển đổi. Đồ án xây dựng và so sánh 3 phương pháp gợi ý khác nhau trên tập dữ liệu hành vi thực tế của một cửa hàng điện tử, sau đó **đóng gói thành 1 REST API phục vụ thời gian thực**, có kiểm thử tự động và triển khai bằng Docker.

---

## Dataset

**Nguồn:** [E-Commerce Events History in Electronics Store — Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store)

| Thông tin | Giá trị |
|-----------|---------|
| Tổng số bản ghi | 11,157 |
| Số người dùng | 10,640 |
| Số sản phẩm | 5,844 |
| Tương tác trung bình | 1.05 hành động/người |

**Các cột dữ liệu:** `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`

**Phân phối hành vi:**
- View (xem): **89.0%**
- Cart (thêm giỏ): **6.7%**
- Purchase (mua): **4.3%**

---

## Pipeline xử lý (huấn luyện mô hình)

```
Raw Data (CSV)
      │
      ▼
GD1: Tiền xử lý
  ├── Chuyển event_time → datetime
  ├── Xử lý trùng lặp (0 dòng trùng)
  ├── Điền 'Unknown' vào category_code & brand bị thiếu
  ├── Cyclic Encoding cho thời gian (sin/cos)
  ├── One-Hot Encoding cho event_type & brand
  └── StandardScaler cho price
      │
      ▼
GD2: Tách dữ liệu (Time-based Split)
  ├── Sắp xếp theo thời gian
  ├── Train: 80% dữ liệu cũ nhất
  ├── Test:  20% dữ liệu mới nhất
  └── Gán trọng số tương tác (View=1, Cart=2, Purchase=3)
      │
      ▼
GD3: Xây dựng mô hình
  ├── Baseline: Trending (Popularity-based)
  ├── Model A: Collaborative Filtering (ALS)
  ├── Model B: Content-Based Filtering (TF-IDF + Cosine Similarity)
  └── Model C: Hybrid (Waterfall Logic)
      │
      ▼
GD4: Đánh giá & So sánh (Precision@10, Recall@10)
      │
      ▼
Export model (.pkl) + log MLflow → artifacts/
```

---

## Các mô hình

### Baseline — Trending (Popularity-based)
Gợi ý Top 10 sản phẩm có tổng điểm tương tác cao nhất cho **tất cả** người dùng. Không cá nhân hóa, dùng làm chuẩn tham chiếu.

### Model A — Collaborative Filtering (ALS)
Sử dụng thư viện `implicit` với thuật toán **Alternating Least Squares**, chuyên xử lý **implicit feedback** (không có rating sao, chỉ có hành vi).

- Ma trận User-Item: **8,534 users × 5,111 products**
- Độ thưa (Sparsity): **99.98%**
- Hyperparameters: `factors=64`, `regularization=0.05`, `iterations=20`, `alpha=40`

### Model B — Content-Based Filtering (TF-IDF)
Vector hóa thuộc tính sản phẩm (`brand` + `category_code`) bằng **TF-IDF**, tính độ tương đồng bằng **Cosine Similarity**.

- Ma trận tương đồng: **5,844 × 5,844**
- Fallback: nếu sản phẩm chưa được học → gợi ý theo cùng danh mục

### Model C — Hybrid (Waterfall Logic)
Phân luồng người dùng thành 3 nhóm, áp dụng chiến lược phù hợp:

| Nhóm | Định nghĩa | Chiến lược | Số lượng |
|------|-----------|------------|----------|
| Khách thân thiết | ≥ 2 tương tác, có trong ALS | **Weighted Hybrid** (ALS 60% + Content 40%) | 322 users |
| Khách vãng lai | < 2 tương tác | **Content-Based** (dựa trên sản phẩm xem gần nhất) | 8,212 users |
| Khách mới (Cold Start) | Chưa có lịch sử | **Trending** (sản phẩm bán chạy nhất) | — |

**Công thức Hybrid:**

$$Score_{final} = (w_{ALS} \times Score_{ALS}) + (w_{Content} \times Score_{Content})$$

Logic này được tái hiện lại nguyên vẹn trong `model_utils.py` để phục vụ real-time qua API (xem phần Kiến trúc bên dưới).

---

## Kết quả đánh giá (Top-10 Recommendation)

Đánh giá trên **21 người dùng active** trong tập test:

| Mô hình | Precision@10 | Recall@10 |
|---------|-------------|-----------|
| **Hybrid (Kết hợp)** | **4.76%** | **33.33%** |
| Content-Based | 4.08% | 26.67% |
| ALS (Collaborative) | 1.90% | 13.33% |
| Trending (Baseline) | 1.43% | 10.00% |

> **Nhận xét:** Hybrid vượt trội nhờ kết hợp được thế mạnh của cả ALS (hiểu hành vi người dùng) và Content-Based (hiểu đặc tính sản phẩm). Precision thấp (~4.7%) là bình thường với bài toán Top-N Recommendation, do tập test chỉ có 21 user active và độ thưa dữ liệu (sparsity) rất cao (99.98%).

Bảng này được log tự động lên **MLflow** mỗi lần train (xem phần bên dưới), và đọc lại được từ Streamlit dashboard.

---

## Kiến trúc & Triển khai (Serving Architecture)

Model sau khi train ở notebook được **export ra `.pkl`** và phục vụ qua 1 REST API độc lập, tách rời hoàn toàn khỏi notebook:

```
                         ┌─────────────────┐
   main.ipynb  ──export──▶   artifacts/    │  (.pkl: ALS, cosine_sim,
  (train models)          │   *.pkl files   │   trending, loyal_user_ids)
                         └────────┬─────────┘
                                  │ mount (volume)
                                  ▼
   PostgreSQL  ◀──SQLAlchemy──  FastAPI (app.py + model_utils.py)
  (feature_engineering.sql:                │
   user_last_viewed_product)               ▼
                                   GET /recommend/{user_id}
                                   GET /health
```

- **`feature_engineering.sql`**: tạo view `user_last_viewed_product`, `user_features`, `trending_top10` trực tiếp trên PostgreSQL — tách phần feature engineering "online" ra khỏi notebook.
- **`model_utils.py`**: load các file `.pkl`, tái hiện đúng logic Hybrid Waterfall, và query PostgreSQL để biết sản phẩm user vừa xem gần nhất.
- **`app.py`**: expose logic trên qua FastAPI (`/recommend/{user_id}?n=10`).
- **`tests/`**: unit test cho toàn bộ logic phân nhóm (loyal/casual/cold-start) và endpoint API, dùng mock nên **không cần model thật hay DB thật để chạy** — chạy được trên CI.
- **`.github/workflows/ci.yml`**: tự động chạy test + build Docker image mỗi lần push.

---

## Cách chạy dự án

### 1. Train model (notebook)

```bash
pip install implicit pandas numpy scikit-learn matplotlib seaborn scipy tqdm
jupyter notebook main.ipynb
```

Chạy tuần tự **GD1 → GD2 → GD3 → GD4**, sau đó chạy 2 cell cuối:
1. `export_models_from_notebook.py` (đã dán sẵn vào notebook) → tạo `artifacts/*.pkl`
2. `mlflow_logging_cell.py` (xem mục MLflow bên dưới) → log kết quả + xuất `evaluation_results.csv`

### 2. Chạy API cục bộ (không Docker)

```bash
pip install -r requirements.txt
# tạo file .env với DATABASE_URL=postgresql://user:pass@localhost:5432/ecommerce
psql -U postgres -d ecommerce -f feature_engineering.sql
uvicorn app:app --reload --port 8000
```

Test: `curl http://localhost:8000/recommend/123` hoặc mở `http://localhost:8000/docs`.

### 3. Chạy toàn bộ bằng Docker Compose (khuyến nghị)

```bash
docker compose up --build
```

Tự động dựng PostgreSQL + chạy `feature_engineering.sql` + build & chạy API — chỉ cần đã có sẵn `artifacts/` (từ bước 1).

### 4. Chạy test

```bash
pip install -r requirements-dev.txt
pytest -v
```

### 5. MLflow — theo dõi thí nghiệm (experiment tracking)

```bash
pip install -r requirements-dashboard.txt
```

Dán nội dung `mlflow_logging_cell.py` vào 1 ô mới trong `main.ipynb` (chạy sau GD4), rồi:

```bash
mlflow ui
```

Mở `http://localhost:5000` — xem lại hyperparameter + Precision/Recall của từng lần train, so sánh trực tiếp giữa các lần chỉnh `factors`, `regularization`,...

### 6. Streamlit dashboard — demo trực quan

```bash
streamlit run streamlit_app.py
```

Giao diện web cho phép:
- Nhập `user_id` → xem ngay gợi ý + nhóm khách hàng (loyal/casual/cold-start) mà không cần gõ `curl`
- Xem biểu đồ so sánh Precision@10/Recall@10 giữa 4 phương pháp (đọc từ `evaluation_results.csv` nếu đã chạy MLflow, không thì dùng số liệu trong bảng kết quả ở trên)

---

## Cấu trúc project

```
E-Commerce-System/
├── main.ipynb                        ← Notebook chính (EDA, train, export, MLflow)
├── data.csv                          ← Dataset (tải từ Kaggle, gitignored)
├── export_models_from_notebook.py    ← Cell export model → artifacts/*.pkl
├── mlflow_logging_cell.py            ← Cell log kết quả train lên MLflow
├── feature_engineering.sql           ← Tạo bảng/view trên PostgreSQL
├── app.py                            ← FastAPI serving
├── model_utils.py                    ← Logic Hybrid Waterfall + load artifacts
├── streamlit_app.py                  ← Dashboard demo
├── tests/
│   ├── test_model_utils.py           ← Unit test logic gợi ý
│   └── test_app.py                   ← Test API endpoints
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml          ← CI: test + build Docker mỗi lần push
├── requirements.txt                  ← Dependency cho API (production)
├── requirements-dev.txt              ← pytest, httpx (test)
├── requirements-dashboard.txt        ← mlflow, streamlit (demo/tracking)
├── LICENSE
└── RECOMMENDATION_SYSTEM.pdf         ← Báo cáo đồ án
```

---

## Thư viện sử dụng

**Notebook/Modeling:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `implicit`, `tqdm`
**API Serving:** `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`
**Test & CI:** `pytest`, `httpx`
**Tracking & Demo:** `mlflow`, `streamlit`

---

## Tài liệu tham khảo

- Bài giảng môn Python cho Khoa học Dữ liệu — ThS. Hà Văn Thảo
- [7 Types of Hybrid Recommendation System](https://medium.com/analytics-vidhya/7-types-of-hybrid-recommendation-system-3e4f78266ad8)
- [Tổng quan về Recommender System](https://viblo.asia/p/tong-quan-ve-recommender-system-recommender-system-co-ban-phan-1-924lJGBb5PM)

---

## License

Phát hành theo giấy phép [MIT](LICENSE).
