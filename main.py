from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from recommend import recommend_books_with_reason
import os
from dotenv import load_dotenv
load_dotenv()

# for start using 'uvicorn main:app --reload'
app = FastAPI()
API_SECRET_KEY = os.getenv("MY_API_KEY")

# CORS 설정 (React 연동 시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 "http://localhost:3000" 등
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerItem(BaseModel):
    question_id: int
    answer: str

class RecommendRequest(BaseModel):
    user_id: str
    answers: List[AnswerItem]

# 데이터 입력 모델
class SurveyRequest(BaseModel):
    user_id: Union[str, None] = None
    answers: dict

# 응답 모델
class BookRecommendation(BaseModel):
    title: str
    author: str
    hashtag: Union[str, None] = None
    reason: Union[str, None] = None  # ⭐ 추천 이유 필드 추가

@app.post("/recommend", response_model=List[BookRecommendation])
def recommend(payload: RecommendRequest, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ")[1]
    if token != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    user_input = [
        item for item in payload.answers
        if item.answer and "-" not in item.answer
    ]

    print(user_input, flush=True)

    recommendations = recommend_books_with_reason(user_input, top_n=6)

    return recommendations

# 헬스 체크용
@app.get("/")
def read_root():
    return {"message": "FastAPI is running on Render!"}
