from pathlib import Path
from yiutils.project_utils import find_project_root

def get_database_paths(profile: str = "local") -> tuple[Path, Path]:
    if profile == "local":
        project_root = find_project_root("docker-compose.yml")
    else:  # docker
        # TODO: config this for docker
        project_root = Path("/app")

    vector_store_db_path = project_root / "data" / "db" / "vector_store.db"
    trait_profile_db_path = (
        project_root / "data" / "db" / "trait_profile_db.db"
    )

    if not vector_store_db_path.exists():
        raise FileNotFoundError(
            f"Vector store database not found at: {vector_store_db_path}"
        )

    if not trait_profile_db_path.exists():
        raise FileNotFoundError(
            f"Trait profile database not found at: {trait_profile_db_path}"
        )
    res = (vector_store_db_path, trait_profile_db_path)
    return res


