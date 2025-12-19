import pytest
from unittest.mock import MagicMock

# ---------------------------
# Fake DB Session
# ---------------------------
@pytest.fixture
def mock_db():
    db = MagicMock()

    fake_lab = MagicMock()
    fake_lab.lab_id = 1
    fake_lab.summary = "This lab focuses on computer vision and deep learning."

    db.query().filter().first.return_value = fake_lab
    return db


# ---------------------------
# Fake Lab Search Result (Milvus)
# ---------------------------
class FakeHit:
    def __init__(self, lab_id, professor, department, similarity):
        self.distance = similarity
        self.entity = {
            "lab_id": lab_id,
            "professor_name": professor,
            "department": department,
        }


@pytest.fixture
def fake_lab_hits():
    return [[
        FakeHit(1, "Prof. A", "CS", 0.8),
        FakeHit(2, "Prof. B", "EE", 0.6),
        FakeHit(3, "Prof. A", "CS", 0.9),  # duplicate professor
        FakeHit(4, "Prof. C", "ME", 0.4),  # below threshold
    ]]
