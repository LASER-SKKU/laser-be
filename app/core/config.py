import yaml
from pathlib import Path

def load_secrets():
    """프로젝트 루트의 secrets.yml 파일을 읽어 설정 반환"""
    secrets_path = Path(__file__).resolve().parent.parent.parent / "secrets.yml"
    with open(secrets_path, "r") as f:
        return yaml.safe_load(f)

secrets = load_secrets()
