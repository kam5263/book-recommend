from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from recommend import recommend_books_with_reason

# for start using 'uvicorn main:app --reload'
app = FastAPI()

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
def recommend(request: RecommendRequest):
    # user_input = list(request.answers.values())
    user_input = [
        item for item in request.answers
        if item.answer and "-" not in item.answer
    ]

    print(user_input)
    # filtered_input = filter_negative_answers(user_input)
    # print(filtered_input)
    recommendations = recommend_books_with_reason(user_input, top_n=6)

    return recommendations

# 헬스 체크용
@app.get("/")
def read_root():
    return {"message": "FastAPI is running on Render!"}
