# app/services/paper_embedding_service.py
from typing import Optional
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from openai import OpenAI
from sqlalchemy.orm import Session
from app.core.config import secrets
from app.models.paper import Paper

MILVUS_HOST = "43.201.113.80"
MILVUS_PORT = "19530"
COLLECTION_NAME = "paper_embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536

client = OpenAI(api_key=secrets["openai"]["api_key"])


def connect_milvus():
    try:
        addr = connections.get_connection_addr("default")
        utility.list_collections()
        return
    except:
        try:
            connections.disconnect("default")
        except:
            pass

        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=10,
        )


def _get_collection_if_exists() -> Optional[Collection]:
    connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    return None


def create_paper_collection_if_not_exists() -> Collection:
    connect_milvus()

    exist = _get_collection_if_exists()
    if exist:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")
        return exist

    fields = [
        FieldSchema(name="paper_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="lab_id", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=450),
        FieldSchema(name="publication_year", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]

    schema = CollectionSchema(fields, description="Paper Embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"✅ Created collection '{COLLECTION_NAME}'.")
    return collection


def insert_paper_embeddings(db: Session, embed_batch_size: int = 100, milvus_batch_size: int = 200):
    """
    - Embedding API를 batch로 처리하여 속도 향상
    - Milvus insert를 batch 단위로 처리
    - insert 실패 시 print로만 알림 (retry 없음)
    """

    connect_milvus()
    collection = create_paper_collection_if_not_exists()
    collection.load()

    # summary가 있는 논문만 처리
    papers = db.query(Paper).filter(Paper.summary.isnot(None)).all()
    if not papers:
        return {"inserted": 0}

    total = len(papers)
    print(f"[Start] Total papers to embed: {total}")

    # Embedding API batch buffer
    text_buffer = []
    paper_buffer = []

    # Milvus insert buffer
    milvus_ids = []
    milvus_lab_ids = []
    milvus_titles = []
    milvus_years = []
    milvus_vecs = []

    embedded_count = 0
    inserted_count = 0

    for paper in papers:

        # batch embedding buffer에 추가
        text_buffer.append(paper.summary.strip())
        paper_buffer.append(paper)

        # -----------------------------------------
        # 🔥 Batch Embedding (100개 단위)
        # -----------------------------------------
        if len(text_buffer) >= embed_batch_size:

            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text_buffer
                )
                embeddings = [d.embedding for d in resp.data]

            except Exception as e:
                print(f"[Embedding Error] Failed batch: {e}")
                text_buffer.clear()
                paper_buffer.clear()
                continue

            # embedding 결과를 milvus batch buffer로 이동
            for i, paper_obj in enumerate(paper_buffer):
                milvus_ids.append(paper_obj.paper_id)
                milvus_lab_ids.append(paper_obj.lab_id)
                milvus_titles.append(paper_obj.title)
                milvus_years.append(paper_obj.publication_year or 0)
                milvus_vecs.append(embeddings[i])

            embedded_count += len(text_buffer)
            print(f"[Embedding Done] {embedded_count}/{total}")

            text_buffer.clear()
            paper_buffer.clear()

            # -----------------------------------------
            # 🔥 Milvus Batch Insert (200개 단위)
            # -----------------------------------------
            if len(milvus_ids) >= milvus_batch_size:
                try:
                    collection.insert([
                        milvus_ids,
                        milvus_lab_ids,
                        milvus_titles,
                        milvus_years,
                        milvus_vecs
                    ])
                    collection.flush()

                    inserted_count += len(milvus_ids)
                    print(f"[Milvus] Inserted batch: {len(milvus_ids)} | Total: {inserted_count}")

                except Exception as e:
                    print(f"[Milvus Error] Batch insert failed: {e}")

                milvus_ids.clear()
                milvus_lab_ids.clear()
                milvus_titles.clear()
                milvus_years.clear()
                milvus_vecs.clear()

    # -----------------------------------------
    # 🔥 마지막 남은 Embedding batch 처리
    # -----------------------------------------
    if text_buffer:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text_buffer
            )
            embeddings = [d.embedding for d in resp.data]
        except Exception as e:
            print(f"[Embedding Error] Final batch failed: {e}")
            return {"inserted": inserted_count}

        for i, paper_obj in enumerate(paper_buffer):
            milvus_ids.append(paper_obj.paper_id)
            milvus_lab_ids.append(paper_obj.lab_id)
            milvus_titles.append(paper_obj.title)
            milvus_years.append(paper_obj.publication_year or 0)
            milvus_vecs.append(embeddings[i])

        embedded_count += len(text_buffer)
        print(f"[Embedding Done] {embedded_count}/{total}")

    # -----------------------------------------
    # 🔥 마지막 Milvus batch insert
    # -----------------------------------------
    if milvus_ids:
        try:
            collection.insert([
                milvus_ids,
                milvus_lab_ids,
                milvus_titles,
                milvus_years,
                milvus_vecs
            ])
            collection.flush()

            inserted_count += len(milvus_ids)
            print(f"[Milvus] Inserted final batch: {len(milvus_ids)}")

        except Exception as e:
            print(f"[Milvus Error] Final batch insert failed: {e}")

    print(f"[Complete] Total inserted: {inserted_count}")
    return {"inserted": inserted_count}


def get_paper_collection_count():
    connect_milvus()
    coll = _get_collection_if_exists()
    if not coll:
        return {"exists": False, "count": 0}
    coll.load()
    return {"exists": True, "count": coll.num_entities}


def drop_paper_collection():
    connect_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        return {"deleted": False, "reason": "not_found"}

    utility.drop_collection(COLLECTION_NAME)
    return {"deleted": True}
