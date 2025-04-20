import pandas as pd
from scipy import sparse
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from typing import List

# 불러오기만 해도 추천 가능
df = pd.read_csv("preprocessed_books.csv", encoding="utf-8-sig")
tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")

# vectorizer도 불러와야 새로운 문장을 추천하려면 사용 가능
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def handle_q8_sijip(item) -> List[str]:
    if item.answer == "시집_자주":
        return ["시집"] * 3
    elif item.answer == "시집_가끔":
        return ["시집"]
    return []

NEGATIVE_KEYWORDS = {"거의", "안", "없다", "싫다", "싫어", "아니다"}
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

def preprocess_korean_text(text):
    okt = Okt()
    if pd.isna(text):
        return ""
    tokens = okt.pos(text, stem=True)
    filtered = [word for word, tag in tokens if tag in ['Noun', 'Adjective'] and word not in NEGATIVE_KEYWORDS]
    return ' '.join(filtered)

def recommend_boods_by_input(user_input, top_n=5):
    # 사용자의 선택을 하나의 문장처럼 묶음
    user_text = " ".join(user_input)

    # 이 텍스트를 형태소 분석하고 TF-IDF 벡터화
    user_vec = vectorizer.transform([preprocess_korean_text(user_text)])
    # 전체 책 벡터들과 유사도 비교
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n+1]

    # 결과 출력
    return df.iloc[top_indices][["title", "author", "hashtag"]]

def recommend_books_with_reason(user_input: List[dict], top_n=5):
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