# app/services/lab_embedding_service.py
from typing import Optional
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from openai import OpenAI
from sqlalchemy.orm import Session
from app.core.config import secrets
from app.models.lab import Lab

# -------------------------------
# Milvus / OpenAI 설정
# -------------------------------
MILVUS_HOST = "43.201.113.80"
MILVUS_PORT = "19530"
COLLECTION_NAME = "lab_embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536

client = OpenAI(api_key=secrets["openai"]["api_key"])


def connect_milvus():
    """Milvus 서버 연결 (없으면 자동 생성, 이미 있으면 무시)"""
    try:
        # 연결 테스트 - 실제로 작동하는지 확인
        addr = connections.get_connection_addr("default")
        # 연결이 있어도 실제로 작동하는지 테스트
        utility.list_collections()  # ← 실제 작동 테스트
        print(f"✅ Already connected to Milvus at {addr}")
        return
    except Exception as e:
        print(f"⚠️ Connection check failed: {e}, reconnecting...")
        # 기존 연결이 있다면 끊기
        try:
            connections.disconnect("default")
        except:
            pass
        
        # 새로 연결
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=10
            )
            print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise


def _get_collection_if_exists() -> Optional[Collection]:
    """컬렉션 존재 시 반환, 없으면 None"""
    connect_milvus()  # 연결 확인
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    return None


def create_lab_collection_if_not_exists() -> Collection:
    """Milvus에 lab_embeddings 컬렉션 생성 (없으면 자동 생성)"""
    connect_milvus()  # 연결 확인
    
    exist = _get_collection_if_exists()
    if exist:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")
        return exist

    fields = [
        FieldSchema(name="lab_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="professor_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields, description="LASER Lab Embeddings")

    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Created new collection '{COLLECTION_NAME}'.")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Created index on 'embedding' field (COSINE metric).")

    return collection


def get_text_embedding(text: str):
    """OpenAI API로 텍스트 임베딩 생성"""
    if not text or not text.strip():
        return None
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text.strip())
        return resp.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


def insert_lab_embeddings(db: Session):
    """
    MySQL에서 summary가 존재하는 연구실을 불러와 Milvus에 임베딩 저장.
    - 컬렉션이 없으면 생성
    - 컬렉션이 있으면 그대로 사용 (덮어쓰기/재생성 없음)
    """
    connect_milvus()  # 명시적 연결
    collection = create_lab_collection_if_not_exists()
    collection.load()

    labs = db.query(Lab).filter(Lab.summary.isnot(None)).all()
    if not labs:
        print("⚠️ No labs found with summaries.")
        return {"inserted": 0}

    inserted = 0
    batch_ids = []
    batch_names = []
    batch_depts = []
    batch_vecs = []

    for lab in labs:
        emb = get_text_embedding(lab.summary)
        if not emb:
            print(f"⚠️ Skipped Lab ID={lab.lab_id} (no embedding)")
            continue
        batch_ids.append(lab.lab_id)
        batch_names.append(lab.professor_name or "")
        batch_depts.append(lab.department or "")
        batch_vecs.append(emb)

    if not batch_ids:
        print("⚠️ No embeddings generated.")
        return {"inserted": 0}

    try:
        collection.insert([batch_ids, batch_names, batch_depts, batch_vecs])
        collection.flush()
        inserted = len(batch_ids)
        print(f"✅ Inserted {inserted} lab embeddings into Milvus.")
        return {"inserted": inserted}
    except Exception as e:
        print(f"[Milvus Insert Error] {e}")
        raise  # 에러를 상위로 전파


def get_lab_collection_count() -> dict:
    """
    lab_embeddings 컬렉션 엔트리 수 반환.
    컬렉션이 없으면 0으로 간주.
    """
    connect_milvus()  # 연결 확인
    coll = _get_collection_if_exists()
    if not coll:
        return {"collection": COLLECTION_NAME, "exists": False, "count": 0}
    coll.load()
    count = coll.num_entities
    return {"collection": COLLECTION_NAME, "exists": True, "count": count}


def drop_lab_collection() -> dict:
    """
    lab_embeddings 컬렉션 삭제 (존재할 때만).
    """
    connect_milvus()  # 연결 확인
    if not utility.has_collection(COLLECTION_NAME):
        return {"collection": COLLECTION_NAME, "deleted": False, "reason": "not_found"}

    try:
        utility.drop_collection(COLLECTION_NAME)
        print(f"🗑️ Dropped collection '{COLLECTION_NAME}'.")
        return {"collection": COLLECTION_NAME, "deleted": True}
    except Exception as e:
        print(f"[Milvus Drop Error] {e}")
        return {"collection": COLLECTION_NAME, "deleted": False, "error": str(e)}