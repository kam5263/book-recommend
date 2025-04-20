import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from kiwipiepy import Kiwi
from typing import List
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 형태소 분석기 초기화 (전역에서 한 번만)
kiwi = Kiwi()

print("[DEBUG] csv 파일 불러오는 중")
df = pd.read_csv(os.path.join(BASE_DIR, "hangle_preprocessed_books_deduple.csv"), encoding="utf-8-sig")

NEGATIVE_KEYWORDS = {"거의", "안", "없다", "싫다", "싫어", "아니다"}
# 형태소 분석기 (Kiwi) 기반 전처리 함수
def preprocess_korean_text(text):
    if pd.isna(text):
        return ""
    tokens = kiwi.tokenize(text)
    filtered = [token.form for token in tokens
                if token.tag in ['NNG', 'NNP', 'VA'] and token.form not in NEGATIVE_KEYWORDS]
    return ' '.join(filtered)

# start_time = time.time()
# print("[DEBUG] clean_text 재생성 중")
# df["clean_text"] = df["clean_text"].apply(preprocess_korean_text)
# end_time = time.time()
# print(f"실행 시간: {end_time - start_time:.4f}초")

start_time = time.time()
# TF-IDF 벡터화 (직접 생성)
print("[DEBUG] TfidfVectorizer 실행 중")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
end_time = time.time()
print(f"실행 시간: {end_time - start_time:.4f}초")

# 특수 핸들러 및 키워드 설정
def handle_q8_sijip(item) -> List[str]:
    if item.answer == "시집_자주":
        return ["시집"] * 3
    elif item.answer == "시집_가끔":
        return ["시집"]
    return []

# Q5 응답 → 강조할 질문 ID 및 가중치 목록
Q5_WEIGHT_MAP = {
    "작가": [(2, 2.0)],
    "내용": [(3, 1.0), (4, 2.0), (7, 3.0), (9, 4.0)],
    "감성": [(1, 2.0), (6, 3.0)],
    "추천": [(10, 2.0)]
}

SPECIAL_HANDLERS = {
    8: handle_q8_sijip
    # 향후 추가 가능
}

def recommend_books_with_reason(user_input: List[dict], top_n=5):
    print("[DEBUG] 추천 요청 시작", flush=True)
    print("[DEBUG] 사용자 입력:", user_input, flush=True)
    # Q5 응답 확인
    q5_answer = next((item.answer for item in user_input if item.question_id == 5), None)
    weight_map = Q5_WEIGHT_MAP.get(q5_answer, [])
    # 질문 ID별 가중치 dict 생성 (없으면 1.0)
    question_weights = {qid: weight for qid, weight in weight_map}

    answer_texts = []
    for item in user_input:
        # Q8 시집 처리
        # 🧠 특수 로직이 있으면 우선 실행
        if item.question_id in SPECIAL_HANDLERS:
            special_keywords = SPECIAL_HANDLERS[item.question_id](item)
            answer_texts.extend(special_keywords)
            continue  # 특수 로직 처리했으니 일반 로직은 스킵
        
        weight = question_weights.get(item.question_id, 1.0)
        repeat_count = round(weight * 1.0)  # → 2.5 → 2번 반복 (정수로)
        answer_texts.extend([item.answer] * repeat_count)

    user_text = " ".join(answer_texts)

    # "answer" 필드만 추출하여 벡터화
    answer_texts = [item.answer for item in user_input]

    user_text = " ".join(answer_texts)
    user_proc = preprocess_korean_text(user_text)
    user_vec = vectorizer.transform([user_proc])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]

    feature_names = vectorizer.get_feature_names_out()
    user_vec_array = user_vec.toarray()[0]

    results = []

    for idx in top_indices:
        book = df.iloc[idx]
        book_vec = tfidf_matrix[idx].toarray()[0]

        # 사용자와 책 모두에서 TF-IDF가 높은 단어 추출
        top_user_keywords = set(
            [feature_names[i] for i in user_vec_array.argsort()[::-1][:20] if user_vec_array[i] > 0]
        )
        top_book_keywords = set(
            [feature_names[i] for i in book_vec.argsort()[::-1][:20] if book_vec[i] > 0]
        )

        overlap = list(top_user_keywords & top_book_keywords)
        overlap = sorted(overlap, key=lambda w: user_vec_array[feature_names.tolist().index(w)], reverse=True)
        reason_keywords = ", ".join(overlap[:3]) if overlap else "공통된 키워드 없음"

        results.append({
            "title": book["title"],
            "author": book["author"],
            "hashtag": book.get("hashtag", ""),
            "reason": f"'{reason_keywords}' 키워드를 바탕으로 추천되었습니다."
        })

    return results