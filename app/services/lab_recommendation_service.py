from pymilvus import connections, Collection, utility
from openai import OpenAI
from app.core.config import secrets
from app.services.lab_embedding_service import connect_milvus
from sqlalchemy.orm import Session
from app.models.lab import Lab

client = OpenAI(api_key=secrets["openai"]["api_key"])

MILVUS_HOST = "43.201.113.80"
MILVUS_PORT = "19530"
LAB_COLLECTION = "lab_embeddings"
PAPER_COLLECTION = "paper_embeddings"  # TODO: 논문 임베딩 추가 후 사용
EMBEDDING_MODEL = "text-embedding-3-small"


# -------------------------------
# 0️⃣ 사용자 입력 정제 (LLM)
# -------------------------------
def normalize_user_query(user_text: str) -> str:
    """
    사용자의 관심사 문장을 LLM으로 간결하고 명확한 영어 연구 관심사 문장으로 정리.
    - 한글/혼합 언어 입력도 지원.
    - 불필요한 문체 제거하고 핵심 연구 키워드 중심으로 변환.
    """
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
    """사용자 입력 텍스트를 OpenAI 임베딩 벡터로 변환"""
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
# 2️⃣ 추천 이유 생성 (LLM)
# -------------------------------
def generate_recommendation_reason(user_text: str, lab_summary: str, similar_papers: list = None):
    """
    LLM으로 사용자 관심사와 연구실 요약을 비교하여 객관적인 분석 결과를 생성.
    - 유사한 부분과 다른 부분을 명확히 구분해서 설명.
    - TODO: 이후 paper_embeddings 추가 시, 유사 논문 제목/개수도 prompt에 포함.
    """
    prompt = f"""
You are an academic matching evaluator.

Compare the user's research interest and the lab summary below.

The goal is to **analyze whether they align or not** — do not assume similarity.
Be objective and factual. If there is overlap, explain specifically what topics or methods are similar.
If they are different or unrelated, state that clearly and briefly explain why.

User research interest:
"{user_text}"

Lab summary:
"{lab_summary}"

{"The following are papers from this lab that may relate to the user's interests:\n" + ', '.join(similar_papers) if similar_papers else ""}

Write your analysis in 2–3 short sentences of clear academic English.
Avoid exaggeration or making up connections that are not stated.
"""
    try:
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
    """
    사용자 관심사 텍스트를 기반으로 연구실 유사도 상위 N개를 반환.

    - Milvus에서 summary 유사도 ≥ threshold 이상 연구실만 필터링
    - TODO: paper_embeddings에서 유사 논문 count를 가져와 가중 평균 점수 계산
    - TODO: 추천 이유 생성 시, 유사 논문 제목 리스트와 count 포함
    """

    connect_milvus()

    if not utility.has_collection(LAB_COLLECTION):
        print(f"❌ '{LAB_COLLECTION}' 컬렉션이 존재하지 않습니다.")
        return []

    lab_collection = Collection(LAB_COLLECTION)
    lab_collection.load()

    # ✅ 사용자 입력 정제 (한글/혼합 입력 → 영어 요약으로)
    normalized_query = normalize_user_query(user_text)

    # ✅ 사용자 입력 임베딩 생성
    query_embedding = get_query_embedding(normalized_query)
    if not query_embedding:
        return []

    # ✅ 연구실 유사도 검색 (limit 넉넉히)
    total_labs = lab_collection.num_entities
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = lab_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=min(1000, total_labs),  # 전체는 너무 많으니 상한 제한
        output_fields=["lab_id", "professor_name", "department"],
    )

    recommendations = []
    for hit in results[0]:
        sim = float(hit.distance)
        if sim < similarity_threshold:
            continue

        # TODO: 이후 lab_id를 기반으로 유사 논문 수 및 제목 가져오기
        # similar_papers = get_similar_papers(lab_id, query_embedding)
        # similar_paper_count = len(similar_papers)
        # normalized_paper_count = similar_paper_count / max_paper_count
        # final_score = (sim * 0.3) + (normalized_paper_count * 0.7)

        recommendations.append({
            "lab_id": hit.entity.get("lab_id"),
            "professor_name": hit.entity.get("professor_name"),
            "department": hit.entity.get("department"),
            "lab_similarity": sim,
            "final_score": sim,  # TODO: 이후 논문 점수 포함해서 수정
            "paper_similarity_count": None,  # TODO
            "recommendation_reason": None,   # 나중에 LLM이 채움
        })

    # ✅ LLM으로 추천 이유 생성 (Top N만)
    for rec in recommendations[:top_k]:
        lab = db.query(Lab).filter(Lab.lab_id == rec["lab_id"]).first()
        lab_summary = (
            lab.summary
            if lab and lab.summary
            else f"Professor {rec['professor_name']}'s lab focuses on {rec['department']} research."
        )

        # TODO: 나중에 유사 논문 리스트(similar_papers)도 넣기
        reason = generate_recommendation_reason(normalized_query, lab_summary)
        rec["recommendation_reason"] = reason


    # ✅ 정렬 후 상위 N개 반환
    recommendations.sort(key=lambda x: x["final_score"], reverse=True)
    return recommendations[:top_k]
