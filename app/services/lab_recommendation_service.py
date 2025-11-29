from pymilvus import connections, Collection, utility
from openai import OpenAI
from app.core.config import secrets
from app.services.lab_embedding_service import connect_milvus
from sqlalchemy.orm import Session
from app.models.lab import Lab

import math

client = OpenAI(api_key=secrets["openai"]["api_key"])

MILVUS_HOST = "43.201.113.80"
MILVUS_PORT = "19530"
LAB_COLLECTION = "lab_embeddings"
PAPER_COLLECTION = "paper_embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"


# -------------------------------
# 0️⃣ 사용자 입력 정제 (LLM)
# -------------------------------
def normalize_user_query(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return user_text

    prompt = f"""
    Rewrite the following text as a concise English research interest description.
    Focus only on research topics, technologies, and goals. Remove filler words.

    --- USER INPUT ---
    {user_text}
    """
    try:
        response = client.chat.completions.create(
            model=secrets["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        normalized = response.choices[0].message.content.strip()
        print(f"[Query Normalized] {normalized}")
        return normalized
    except Exception as e:
        print(f"[LLM ERROR: Query Normalization] {e}")
        return user_text


# -------------------------------
# 1️⃣ 사용자 입력 임베딩 생성
# -------------------------------
def get_query_embedding(user_text: str):
    if not user_text or not user_text.strip():
        return None
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=user_text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


# -------------------------------
# ⭐ 논문 유사도 검색 (추가)
# -------------------------------
def search_similar_papers(query_embedding, limit=500, threshold=0.5):
    """Milvus에서 사용자 관심사와 유사한 논문들 검색"""
    if not utility.has_collection(PAPER_COLLECTION):
        return []

    paper_collection = Collection(PAPER_COLLECTION)
    paper_collection.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = paper_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["paper_id", "lab_id", "title", "publication_year"],
    )

    similar = []
    for hit in results[0]:
        sim = float(hit.distance)
        if sim >= threshold:
            similar.append({
                "paper_id": hit.entity.get("paper_id"),
                "lab_id": hit.entity.get("lab_id"),
                "title": hit.entity.get("title"),
                "similarity": sim,
            })

    return similar


# -------------------------------
# 2️⃣ 추천 이유 생성 (LLM)
# -------------------------------
def generate_recommendation_reason(user_text: str, lab_summary: str, similar_papers: list = None):
    """
    LLM으로 사용자 관심사와 연구실 요약을 비교하여 객관적인 분석 결과를 생성.
    - 너의 기존 prompt 완전 유지
    - similar_papers는 옵션으로 깔끔하게 추가
    """

    prompt = f"""
You are an academic matching evaluator.

Compare the user's research interest and the lab summary below.

The goal is to **analyze whether they align or not** — do not assume similarity.
Be objective and factual. If there is overlap, explain specifically what topics or methods are similar.
If they are different or unrelated, state that clearly and briefly explain why.

User research interest:
\"{user_text}\"

Lab summary:
\"{lab_summary}\"

Write your analysis in 2–3 short sentences of clear academic English.
Avoid exaggeration or making up connections that are not stated.
"""

    try:
        # print(prompt)
        response = client.chat.completions.create(
            model=secrets["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM ERROR: Recommendation Reason] {e}")
        return None


# -------------------------------
# 3️⃣ 연구실 추천 계산
# -------------------------------
def recommend_labs(db: Session, user_text: str, similarity_threshold: float = 0.5, top_k: int = 5):

    connect_milvus()

    if not utility.has_collection(LAB_COLLECTION):
        print(f"❌ '{LAB_COLLECTION}' 컬렉션이 존재하지 않습니다.")
        return []

    lab_collection = Collection(LAB_COLLECTION)
    lab_collection.load()

    # 사용자 입력 정제
    normalized_query = normalize_user_query(user_text)

    # 사용자 입력 임베딩 생성
    query_embedding = get_query_embedding(normalized_query)
    if not query_embedding:
        return []

    # 연구실 summary 검색
    total_labs = lab_collection.num_entities
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = lab_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=min(1000, total_labs),
        output_fields=["lab_id", "professor_name", "department"],
    )

    # ⭐ log-normalization용 max_count 구하기
    max_count = max((len(v) for v in paper_by_lab.values()), default=1)
    max_log = math.log(1 + max_count)

    # ------------------------------------
    # 연구실별 추천 결과 구축
    # ------------------------------------
    recommendations = []

    for hit in results[0]:
        sim = float(hit.distance)
        if sim < similarity_threshold:
            continue

        lab_id = hit.entity.get("lab_id")
        paper_sims = paper_by_lab.get(lab_id, [])

        # --------------------------
        # ⭐ paper top-k score
        # --------------------------
        k = 3
        top_k_sims = sorted(paper_sims, reverse=True)[:k]
        paper_topk_score = sum(top_k_sims) / len(top_k_sims) if top_k_sims else 0.0

        # --------------------------
        # ⭐ paper log count score
        # --------------------------
        paper_count = len(paper_sims)
        if paper_count > 0:
            raw_log = math.log(1 + paper_count)
            paper_count_score = raw_log / max_log
        else:
            paper_count_score = 0.0

        # --------------------------
        # ⭐ 최종 점수 계산
        # --------------------------
        final_score = (
            0.5 * paper_topk_score +
            0.2 * paper_count_score +
            0.3 * sim
        )

        recommendations.append({
            "lab_id": lab_id,
            "professor_name": hit.entity.get("professor_name"),
            "department": hit.entity.get("department"),
            "lab_similarity": sim,
            "paper_topk_score": paper_topk_score,
            "paper_count_score": paper_count_score,
            "final_score": final_score,
            "recommendation_reason": None,
        })

    # ------------------------------------
    # 1) 먼저 점수 기준으로 정렬
    # ------------------------------------
    recommendations.sort(key=lambda x: x["final_score"], reverse=True)
    
    unique_labs = []
    seen_professors = set()

    for rec in recommendations:
        professor = rec["professor_name"]
        if professor in seen_professors:
            continue
        seen_professors.add(professor)
        unique_labs.append(rec)

    # 최종 상위 N개 선택
    top_recs = unique_labs[:top_k]

    # ------------------------------------
    # 2) LLM 추천 이유 생성 (Top N에 대해서만)
    # ------------------------------------
    for rec in top_recs:
        lab = db.query(Lab).filter(Lab.lab_id == rec["lab_id"]).first()

        lab_summary = (
            lab.summary
            if lab and lab.summary
            else f"Professor {rec['professor_name']}'s lab focuses on {rec['department']} research."
        )

        top_papers = sorted(
            [p for p in similar_papers_global if p["lab_id"] == rec["lab_id"]],
            key=lambda x: x["similarity"],
            reverse=True
        )[:3]

        rec["top_similar_papers"] = top_papers

        similar_titles = [p["title"] for p in top_papers]

        reason = generate_recommendation_reason(
            normalized_query,
            lab_summary,
            similar_papers=similar_titles
        )

        rec["recommendation_reason"] = reason

    return top_recs
