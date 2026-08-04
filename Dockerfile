FROM python:3.11-slim

WORKDIR /app

# build-essential can duoc mot so package (scipy, implicit) khi khong co wheel san
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model_utils.py ./

# artifacts/ (file .pkl) KHONG duoc COPY vao image - mount qua volume luc
# chay (xem docker-compose.yml). Ly do: model co the duoc train/export lai
# thuong xuyen, khong nen phai build lai image moi lan co model moi.

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
