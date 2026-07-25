from fastapi import FastAPI, HTTPException
from model_utils import RecommenderService

app = FastAPI(title="E-Commerce Recommendation API", version="0.1.0")

# Load model 1 lần lúc khởi động server (không load lại mỗi request)
recommender = RecommenderService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    try:
        return recommender.recommend(user_id=user_id, n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chạy: uvicorn app:app --reload --port 8000
# Test:  curl http://localhost:8000/recommend/123
