from unittest.mock import patch, MagicMock

from app.services.paper_embedding_service import (
    insert_paper_embeddings,
    get_paper_collection_count,
    drop_paper_collection,
)


# insert_paper_embeddings - 정상 케이스
@patch("app.services.paper_embedding_service.client")
@patch("app.services.paper_embedding_service.create_paper_collection_if_not_exists")
@patch("app.services.paper_embedding_service.connect_milvus")
def test_insert_paper_embeddings_success(
    mock_connect,
    mock_create_collection,
    mock_client,
    mock_db,
):
    # Fake embedding response
    fake_resp = MagicMock()
    fake_resp.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]
    mock_client.embeddings.create.return_value = fake_resp

    # Fake Milvus collection
    fake_collection = MagicMock()
    mock_create_collection.return_value = fake_collection

    # Fake DB papers
    paper1 = MagicMock(
        paper_id=1,
        lab_id=10,
        title="Paper A",
        publication_year=2022,
        summary="AI research",
    )
    paper2 = MagicMock(
        paper_id=2,
        lab_id=20,
        title="Paper B",
        publication_year=2023,
        summary="ML research",
    )

    mock_db.query().filter().all.return_value = [paper1, paper2]

    # 실행
    result = insert_paper_embeddings(
        db=mock_db,
        embed_batch_size=2,
        milvus_batch_size=10,
    )

    # 검증
    mock_client.embeddings.create.assert_called()
    fake_collection.insert.assert_called()
    fake_collection.flush.assert_called()
    assert result["inserted"] == 2


# insert_paper_embeddings - paper 없음
@patch("app.services.paper_embedding_service.create_paper_collection_if_not_exists")
@patch("app.services.paper_embedding_service.connect_milvus")
def test_insert_paper_embeddings_no_papers(
    mock_connect,
    mock_create_collection,
    mock_db,
):
    mock_db.query().filter().all.return_value = []

    result = insert_paper_embeddings(mock_db)

    assert result["inserted"] == 0


# get_paper_collection_count - 컬렉션 없음
@patch("app.services.paper_embedding_service._get_collection_if_exists")
@patch("app.services.paper_embedding_service.connect_milvus")
def test_get_paper_collection_count_not_exists(
    mock_connect,
    mock_get_collection,
):
    mock_get_collection.return_value = None

    result = get_paper_collection_count()

    assert result["exists"] is False
    assert result["count"] == 0


# drop_paper_collection - not found
@patch("app.services.paper_embedding_service.utility.has_collection")
@patch("app.services.paper_embedding_service.connect_milvus")
def test_drop_paper_collection_not_found(
    mock_connect,
    mock_has_collection,
):
    mock_has_collection.return_value = False

    result = drop_paper_collection()

    assert result["deleted"] is False
    assert result["reason"] == "not_found"
