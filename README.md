# Hệ Thống Gợi Ý Sản Phẩm 🛒

Đồ án môn **Python cho Khoa học Dữ liệu** — Lớp 23TTH, Khoa Toán - Tin học  
Giảng viên: ThS. Hà Văn Thảo

---

## Giới thiệu

Trong bối cảnh quá tải thông tin của thương mại điện tử, hệ thống gợi ý đóng vai trò cá nhân hóa trải nghiệm mua sắm và tối ưu tỷ lệ chuyển đổi. Đồ án xây dựng và so sánh 3 phương pháp gợi ý khác nhau trên tập dữ liệu hành vi thực tế của một cửa hàng điện tử.

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

## Pipeline xử lý

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

---

## Kết quả đánh giá (Top-10 Recommendation)

Đánh giá trên **21 người dùng active** trong tập test:

| Mô hình | Precision@10 | Recall@10 |
|---------|-------------|-----------|
| **Hybrid (Kết hợp)** | **4.76%** | **33.33%** |
| Content-Based | 4.08% | 26.67% |
| ALS (Collaborative) | 1.90% | 13.33% |
| Trending (Baseline) | 1.43% | 10.00% |

> **Nhận xét:** Hybrid vượt trội nhờ kết hợp được thế mạnh của cả ALS (hiểu hành vi người dùng) và Content-Based (hiểu đặc tính sản phẩm). Precision thấp (~4.7%) là bình thường với bài toán Top-N Recommendation do ưu tiên độ đa dạng.

---

## Cài đặt

```bash
pip install implicit pandas numpy scikit-learn matplotlib seaborn scipy tqdm
```

**Chạy notebook:**

```bash
jupyter notebook main.ipynb
```

Chạy tuần tự các section: **GD1 → GD2 → GD3 → GD4**

---

## Cấu trúc project

```
RECOMMENDATION_SYSTEM/
├── main.ipynb       ← Notebook chính (đầy đủ 4 giai đoạn)
├── data.csv         ← Dataset (tải từ Kaggle)
└── RECOMMENDATION_SYSTEM.pdf  ← Báo cáo đồ án
```

---

## Thư viện sử dụng

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `implicit`, `tqdm`

---

## Tài liệu tham khảo

- Bài giảng môn Python cho Khoa học Dữ liệu — ThS. Hà Văn Thảo
- [7 Types of Hybrid Recommendation System](https://medium.com/analytics-vidhya/7-types-of-hybrid-recommendation-system-3e4f78266ad8)
- [Tổng quan về Recommender System](https://viblo.asia/p/tong-quan-ve-recommender-system-recommender-system-co-ban-phan-1-924lJGBb5PM)
