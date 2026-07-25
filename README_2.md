# Bước 1: SQL feature engineering + FastAPI serving

## Bạn cần thêm gì, ở đâu — chỉ 3 chỗ

### 1. `export_models_from_notebook.py` — chỗ QUAN TRỌNG NHẤT cần sửa
Mở file này, đối chiếu với các biến thật trong `main.ipynb` của bạn ở
phần GD3 (Xây dựng mô hình). Mỗi dòng `pickle.dump(...)` đang dùng tên
biến giả định theo README cũ (`model_a`, `user_item_csr`,
`cosine_sim_matrix`, `trending_top10_list`, `loyal_user_ids`...).
**Bạn cần đổi các tên này khớp đúng với tên biến thật trong notebook
của mình.** Sau đó copy toàn bộ nội dung file vào ô cuối cùng của
`main.ipynb` (sau khi đã chạy xong GD3) và chạy 1 lần.

→ Kết quả: một thư mục `artifacts/` chứa các file `.pkl`.

### 2. `model_utils.py` — chỗ cần nối PostgreSQL thật
Có 1 dòng đánh dấu `TODO` trong `__init__`:
```python
self.user_last_viewed = {}  # TODO: query PostgreSQL thật
```
Sau khi chạy `feature_engineering.sql` để tạo view `user_last_viewed_product`,
thay dòng đó bằng query thật, ví dụ dùng SQLAlchemy:
```python
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost:5432/ecommerce")
df = pd.read_sql("SELECT user_id, product_id FROM user_last_viewed_product", engine)
self.user_last_viewed = dict(zip(df.user_id, df.product_id))
```

### 3. `feature_engineering.sql` — chạy trước, không cần sửa gì
Chạy trực tiếp trên PostgreSQL sau khi import `data.csv` vào bảng
`interactions` (câu lệnh `\copy` có ghi sẵn ở đầu file).

## Thứ tự chạy

1. Import `data.csv` vào PostgreSQL → chạy `feature_engineering.sql`
2. Sửa và chạy `export_models_from_notebook.py` bên trong `main.ipynb`
   → có thư mục `artifacts/`
3. Copy `artifacts/` vào cùng cấp với `app.py`
4. Sửa `model_utils.py` (mục 2 ở trên) để nối PostgreSQL thật
5. `pip install -r requirements.txt`
6. `uvicorn app:app --reload --port 8000`
7. Test: `curl http://localhost:8000/recommend/123`

## Bước tiếp theo (chưa làm trong bước này)
- Dockerfile để đóng gói `app.py`
- MLflow tracking khi train lại model
- Streamlit dashboard đọc metric từ MLflow
