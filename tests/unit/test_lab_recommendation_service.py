import math
from unittest.mock import patch, MagicMock

from app.services.lab_recommendation_service import recommend_labs


# 핵심 테스트
@patch("app.services.lab_recommendation_service.generate_recommendation_reason")
@patch("app.services.lab_recommendation_service.search_similar_papers")
@patch("app.services.lab_recommendation_service.get_query_embedding")
@patch("app.services.lab_recommendation_service.normalize_user_query")
@patch("app.services.lab_recommendation_service.Collection")
@patch("app.services.lab_recommendation_service.utility.has_collection")
@patch("app.services.lab_recommendation_service.connect_milvus")
def test_recommend_labs_basic_flow(
    mock_connect,
    mock_has_collection,
    mock_collection,
    mock_normalize,
    mock_embedding,
    mock_search_papers,
    mock_reason,
    mock_db,
    fake_lab_hits,
):
    """
    목적:
    - 점수 계산이 정상 동작하는지
    - threshold, 정렬, 중복 제거 확인
    """

    # Mock 설정
    mock_has_collection.return_value = True
    mock_normalize.return_value = "computer vision"
    mock_embedding.return_value = [0.1] * 1536
    mock_reason.return_value = "The research interests align well."

    # Fake Milvus Lab Search
    fake_lab_collection = MagicMock()
    fake_lab_collection.num_entities = 10
    fake_lab_collection.search.return_value = fake_lab_hits
    mock_collection.return_value = fake_lab_collection

    # Fake paper similarity search
    mock_search_papers.return_value = [
        {"lab_id": 1, "similarity": 0.9, "title": "Paper A"},
        {"lab_id": 1, "similarity": 0.7, "title": "Paper B"},
        {"lab_id": 2, "similarity": 0.6, "title": "Paper C"},
        {"lab_id": 3, "similarity": 0.95, "title": "Paper D"},
    ]

    # 실행
    results = recommend_labs(
        db=mock_db,
        user_text="I like AI and vision",
        similarity_threshold=0.5,
        top_k=3,
    )

    # 검증
    # threshold (0.4는 제외)
    lab_ids = [r["lab_id"] for r in results]
    assert 4 not in lab_ids

    # 교수 중복 제거 (Prof. A 하나만)
    professors = [r["professor_name"] for r in results]
    assert professors.count("Prof. A") == 1

    # 정렬 확인 (final_score 내림차순)
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # final_score 범위
    for r in results:
        assert 0.0 <= r["final_score"] <= 1.0

    # 추천 이유 생성 확인
    assert all(r["recommendation_reason"] is not None for r in results)
