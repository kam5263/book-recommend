from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from recommend import recommend_books_with_reason
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SERVICE_ROLE = os.getenv("SERVICE_ROLE")

supabase: Client = create_client(SUPABASE_URL, SERVICE_ROLE)

# for start using 'uvicorn main:app --reload'
app = FastAPI()
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

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
    
    # Supabase: user_answers 테이블에 저장
    supabase.table("user_answers").insert([
        {
            "user_id": payload.user_id,
            "question_id": item.question_id,
            "answers": item.answer
        } for item in payload.answers
    ]).execute()
    
    user_input = [
        item for item in payload.answers
        if item.answer and "-" not in item.answer
    ]

    print(user_input, flush=True)

    recommendations = recommend_books_with_reason(user_input, top_n=6)
    
    # Supabase: recommend_results 테이블에 저장
    supabase.table("recommend_results").insert([
        {
            "user_id": payload.user_id,
            "title": rec["title"],
            "author": rec["author"],
            "hashtag": rec["hashtag"],
            "reason": rec["reason"]
        } for rec in recommendations
    ]).execute()

    return recommendations

@app.get("/popular")
def get_popular_books(request:Request, limit: int = Query(10, description="가져올 상위 도서 개수")):
    """
    popularityScore 기준으로 상위 인기 도서 n개 조회
    """

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = auth_header.split(" ")[1]

    if token != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        response = supabase.table("aladin_books")\
            .select("*")\
            .order("popularityScore", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            return {"count": len(response.data), "results": response.data}
        else:
            return {"count": 0, "results": []}
    except Exception as e:
        return {"error": str(e)}
@app.get("/expensive")
def get_expensive_books(request:Request, limit: int = Query(10, description="비싸고 두꺼운 책")):
    """
    
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = auth_header.split(" ")[1]

    if token != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        response = supabase.table("aladin_books")\
            .select("*")\
            .order("priceStandard", desc=True)\
            .order("itemPage", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            return {"count": len(response.data), "results": response.data}
        else:
            return {"count": 0, "results": []}
    except Exception as e:
        return {"error": str(e)}
@app.get("/thick")
def get_thick_books(request:Request, limit: int = Query(10, description="비싸고 두꺼운 책")):
    """
    
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = auth_header.split(" ")[1]

    if token != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        response = supabase.table("aladin_books")\
            .select("*")\
            .order("itemPage", desc=True)\
            .order("priceStandard", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            return {"count": len(response.data), "results": response.data}
        else:
            return {"count": 0, "results": []}
    except Exception as e:
        return {"error": str(e)}
# 헬스 체크용
@app.get("/")
def read_root():
    return {"message": "FastAPI is running on Render!"}
