from unittest.mock import patch, MagicMock

from app.services.lab_embedding_service import (
    get_text_embedding,
    insert_lab_embeddings,
    get_lab_collection_count
)


# get_text_embedding
@patch("app.services.lab_embedding_service.client")
def test_get_text_embedding_success(mock_client):
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = fake_response

    emb = get_text_embedding("Computer vision research")

    assert emb is not None
    assert len(emb) == 1536


@patch("app.services.lab_embedding_service.client")
def test_get_text_embedding_empty(mock_client):
    emb = get_text_embedding("")
    assert emb is None
    mock_client.embeddings.create.assert_not_called()


# insert_lab_embeddings
@patch("app.services.lab_embedding_service.get_text_embedding")
@patch("app.services.lab_embedding_service.create_lab_collection_if_not_exists")
@patch("app.services.lab_embedding_service.connect_milvus")
def test_insert_lab_embeddings_success(
    mock_connect,
    mock_create_collection,
    mock_get_embedding,
    mock_db,
):
    # Fake embedding
    mock_get_embedding.return_value = [0.2] * 1536

    # Fake Milvus collection
    fake_collection = MagicMock()
    mock_create_collection.return_value = fake_collection

    # Fake DB lab
    fake_lab = MagicMock()
    fake_lab.lab_id = 1
    fake_lab.professor_name = "Prof. A"
    fake_lab.department = "CS"
    fake_lab.summary = "AI and vision"

    mock_db.query().filter().all.return_value = [fake_lab]

    result = insert_lab_embeddings(mock_db)

    fake_collection.insert.assert_called_once()
    fake_collection.flush.assert_called_once()
    assert result["inserted"] == 1


# get_lab_collection_count
@patch("app.services.lab_embedding_service._get_collection_if_exists")
@patch("app.services.lab_embedding_service.connect_milvus")
def test_get_lab_collection_count_not_exists(
    mock_connect,
    mock_get_collection
):
    mock_get_collection.return_value = None

    result = get_lab_collection_count()

    assert result["exists"] is False
    assert result["count"] == 0
