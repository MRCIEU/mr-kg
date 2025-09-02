"""Comprehensive tests for database integration layer."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pytest

from app.core.database import (
    DatabaseConfig,
    DatabaseConnectionPool,
    _resolve_database_paths,
    create_database_config,
)
from app.core.schema_validation import (
    DatabaseHealthChecker,
    SchemaValidator,
)
from app.models.database import (
    EFOEmbedding,
    ModelResult,
    PaginationParams,
    QueryCombination,
    TraitEmbedding,
)
from app.services.repositories import (
    BaseRepository,
    StudyRepository,
    TraitRepository,
)


class TestDatabaseConfig:
    """Test database configuration functionality."""

    def test_create_database_config_local(self):
        """Test creating database config for local profile."""
        config = create_database_config(profile="local")

        assert config.profile == "local"
        assert config.max_connections == 10
        assert config.connection_timeout == 30.0
        assert isinstance(config.vector_store_path, Path)
        assert isinstance(config.trait_profile_path, Path)

    def test_create_database_config_docker(self):
        """Test creating database config for Docker profile."""
        config = create_database_config(profile="docker")

        assert config.profile == "docker"
        assert config.vector_store_path == Path("/app/data/db/vector_store.db")
        assert config.trait_profile_path == Path(
            "/app/data/db/trait_profile_db.db"
        )

    def test_resolve_database_paths_local(self):
        """Test resolving database paths for local profile."""
        vector_path, trait_path = _resolve_database_paths("local")

        assert isinstance(vector_path, Path)
        assert isinstance(trait_path, Path)
        assert "vector_store.db" in str(vector_path)
        assert "trait_profile_db.db" in str(trait_path)

    def test_resolve_database_paths_docker(self):
        """Test resolving database paths for Docker profile."""
        vector_path, trait_path = _resolve_database_paths("docker")

        assert vector_path == Path("/app/data/db/vector_store.db")
        assert trait_path == Path("/app/data/db/trait_profile_db.db")


class TestDatabaseConnectionPool:
    """Test database connection pool functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock database config with temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_db = Path(temp_dir) / "vector_store.db"
            trait_db = Path(temp_dir) / "trait_profile_db.db"

            # Create empty database files
            vector_db.touch()
            trait_db.touch()

            yield DatabaseConfig(
                profile="test",
                vector_store_path=vector_db,
                trait_profile_path=trait_db,
                max_connections=3,
            )

    @pytest.mark.asyncio
    async def test_pool_initialization(self, mock_config):
        """Test connection pool initialization."""
        pool = DatabaseConnectionPool(mock_config)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_conn.execute.return_value.fetchone.return_value = [1]
            mock_connect.return_value = mock_conn

            await pool.initialize()

            assert pool._initialized
            assert len(pool._vector_store_connections) > 0
            assert len(pool._trait_profile_connections) > 0

    @pytest.mark.asyncio
    async def test_get_connection_context_manager(self, mock_config):
        """Test connection context manager functionality."""
        pool = DatabaseConnectionPool(mock_config)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_conn.execute.return_value.fetchone.return_value = [1]
            mock_connect.return_value = mock_conn

            await pool.initialize()

            async with pool.get_vector_store_connection() as conn:
                assert conn is not None

    @pytest.mark.asyncio
    async def test_pool_status(self, mock_config):
        """Test getting pool status information."""
        pool = DatabaseConnectionPool(mock_config)

        status = await pool.get_pool_status()

        assert "vector_store_connections" in status
        assert "trait_profile_connections" in status
        assert "max_connections" in status
        assert "initialized" in status

    @pytest.mark.asyncio
    async def test_connection_pool_error_handling(self, mock_config):
        """Test error handling in connection pool."""
        # Create config with non-existent files
        bad_config = DatabaseConfig(
            profile="test",
            vector_store_path=Path("/nonexistent/vector.db"),
            trait_profile_path=Path("/nonexistent/trait.db"),
            max_connections=3,
        )

        pool = DatabaseConnectionPool(bad_config)

        with pytest.raises(FileNotFoundError):
            await pool.initialize()


class TestSchemaValidator:
    """Test schema validation functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchone.return_value = [1]
        mock_conn.execute.return_value.fetchall.return_value = [
            (0, "trait_index", "INTEGER", 0, None, 1),
            (1, "trait_label", "VARCHAR", 0, None, 0),
            (2, "vector", "FLOAT[]", 0, None, 0),
        ]
        return mock_conn

    def test_validate_table_exists(self, mock_connection):
        """Test validating that a table exists."""
        # Mock table exists
        mock_connection.execute.return_value.fetchone.return_value = [
            "trait_embeddings"
        ]

        validator = SchemaValidator(mock_connection)

        # Mock column definition for test
        mock_columns = []
        table_info = validator._validate_table("trait_embeddings", mock_columns)

        assert table_info.name == "trait_embeddings"
        assert table_info.exists

    def test_validate_table_not_exists(self, mock_connection):
        """Test validating that a table doesn't exist."""
        # Mock table doesn't exist
        mock_connection.execute.return_value.fetchone.return_value = None

        validator = SchemaValidator(mock_connection)

        mock_columns = []
        table_info = validator._validate_table(
            "nonexistent_table", mock_columns
        )

        assert table_info.name == "nonexistent_table"
        assert not table_info.exists
        assert "does not exist" in table_info.error

    def test_view_exists(self, mock_connection):
        """Test checking if a view exists."""
        # Mock view exists
        mock_connection.execute.return_value.fetchone.return_value = [
            "test_view"
        ]

        validator = SchemaValidator(mock_connection)

        assert validator._view_exists("test_view")

    def test_index_exists(self, mock_connection):
        """Test checking if an index exists."""
        # Mock index exists
        mock_connection.execute.return_value.fetchone.return_value = [
            "test_index"
        ]

        validator = SchemaValidator(mock_connection)

        assert validator._index_exists("test_index")


class TestBaseRepository:
    """Test base repository functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock_conn = Mock()
        return mock_conn

    def test_execute_query(self, mock_connection):
        """Test executing a query."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "test")
        ]

        repo = BaseRepository(mock_connection)
        results = repo.execute_query("SELECT * FROM test")

        assert results == [(1, "test")]
        mock_connection.execute.assert_called_once_with("SELECT * FROM test")

    def test_execute_query_with_params(self, mock_connection):
        """Test executing a query with parameters."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "test")
        ]

        repo = BaseRepository(mock_connection)
        results = repo.execute_query("SELECT * FROM test WHERE id = ?", (1,))

        assert results == [(1, "test")]
        mock_connection.execute.assert_called_once_with(
            "SELECT * FROM test WHERE id = ?", (1,)
        )

    def test_execute_one(self, mock_connection):
        """Test executing a query and getting one result."""
        mock_connection.execute.return_value.fetchone.return_value = (1, "test")

        repo = BaseRepository(mock_connection)
        result = repo.execute_one("SELECT * FROM test WHERE id = 1")

        assert result == (1, "test")

    def test_get_count(self, mock_connection):
        """Test getting count of rows."""
        mock_connection.execute.return_value.fetchone.return_value = (42,)

        repo = BaseRepository(mock_connection)
        count = repo.get_count("test_table")

        assert count == 42
        mock_connection.execute.assert_called_once_with(
            "SELECT COUNT(*) FROM test_table"
        )


class TestTraitRepository:
    """Test trait repository functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock_conn = Mock()
        return mock_conn

    @pytest.fixture
    def trait_repo(self, mock_connection):
        """Create trait repository instance."""
        return TraitRepository(mock_connection)

    def test_get_trait_by_index(self, trait_repo, mock_connection):
        """Test getting trait by index."""
        mock_connection.execute.return_value.fetchone.return_value = (
            1,
            "blood pressure",
            [0.1, 0.2, 0.3],
        )

        trait = trait_repo.get_trait_by_index(1)

        assert trait is not None
        assert trait.trait_index == 1
        assert trait.trait_label == "blood pressure"
        assert trait.vector == [0.1, 0.2, 0.3]

    def test_get_trait_by_index_not_found(self, trait_repo, mock_connection):
        """Test getting trait by index when not found."""
        mock_connection.execute.return_value.fetchone.return_value = None

        trait = trait_repo.get_trait_by_index(999)

        assert trait is None

    def test_search_traits(self, trait_repo, mock_connection):
        """Test searching traits by text."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "blood pressure", [0.1, 0.2]),
            (2, "blood glucose", [0.2, 0.3]),
        ]

        traits = trait_repo.search_traits("blood", limit=10, offset=0)

        assert len(traits) == 2
        assert traits[0].trait_label == "blood pressure"
        assert traits[1].trait_label == "blood glucose"

    def test_get_traits_by_indices(self, trait_repo, mock_connection):
        """Test getting multiple traits by indices."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "trait1", [0.1]),
            (2, "trait2", [0.2]),
        ]

        traits = trait_repo.get_traits_by_indices([1, 2])

        assert len(traits) == 2
        assert traits[0].trait_index == 1
        assert traits[1].trait_index == 2

    def test_get_traits_by_empty_indices(self, trait_repo):
        """Test getting traits with empty indices list."""
        traits = trait_repo.get_traits_by_indices([])

        assert traits == []


class TestStudyRepository:
    """Test study repository functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock_conn = Mock()
        return mock_conn

    @pytest.fixture
    def study_repo(self, mock_connection):
        """Create study repository instance."""
        return StudyRepository(mock_connection)

    def test_get_study_by_id(self, study_repo, mock_connection):
        """Test getting study by ID."""
        mock_connection.execute.return_value.fetchone.return_value = (
            1,
            "gpt-4",
            "12345",
            {"exposures": []},
            {"results": []},
        )

        study = study_repo.get_study_by_id(1)

        assert study is not None
        assert study.id == 1
        assert study.model == "gpt-4"
        assert study.pmid == "12345"

    def test_get_studies_by_pmid(self, study_repo, mock_connection):
        """Test getting studies by PMID."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "gpt-4", "12345", {}, {}),
            (2, "deepseek-r1", "12345", {}, {}),
        ]

        studies = study_repo.get_studies_by_pmid("12345")

        assert len(studies) == 2
        assert studies[0].model == "gpt-4"
        assert studies[1].model == "deepseek-r1"

    def test_search_studies_with_filters(self, study_repo, mock_connection):
        """Test searching studies with filters."""
        mock_connection.execute.return_value.fetchall.return_value = [
            (1, "gpt-4", "12345", {}, {})
        ]

        studies = study_repo.search_studies(
            query="diabetes", models=["gpt-4"], limit=10, offset=0
        )

        assert len(studies) == 1
        assert studies[0].model == "gpt-4"


class TestPaginationParams:
    """Test pagination parameters."""

    def test_pagination_offset_calculation(self):
        """Test offset calculation for pagination."""
        pagination = PaginationParams(page=3, page_size=20)

        assert pagination.offset == 40  # (3-1) * 20

    def test_pagination_first_page(self):
        """Test pagination for first page."""
        pagination = PaginationParams(page=1, page_size=10)

        assert pagination.offset == 0

    def test_pagination_validation(self):
        """Test pagination parameter validation."""
        # Valid parameters
        pagination = PaginationParams(page=1, page_size=50)
        assert pagination.page == 1
        assert pagination.page_size == 50

        # Test with edge values
        pagination = PaginationParams(page=1, page_size=1)
        assert pagination.page_size == 1

        pagination = PaginationParams(page=1, page_size=1000)
        assert pagination.page_size == 1000


class TestModelInstantiation:
    """Test model instantiation and validation."""

    def test_trait_embedding_model(self):
        """Test TraitEmbedding model creation."""
        trait = TraitEmbedding(
            trait_index=1, trait_label="blood pressure", vector=[0.1, 0.2, 0.3]
        )

        assert trait.trait_index == 1
        assert trait.trait_label == "blood pressure"
        assert trait.vector == [0.1, 0.2, 0.3]

    def test_trait_embedding_without_vector(self):
        """Test TraitEmbedding model without vector."""
        trait = TraitEmbedding(trait_index=1, trait_label="blood pressure")

        assert trait.trait_index == 1
        assert trait.trait_label == "blood pressure"
        assert trait.vector is None

    def test_efo_embedding_model(self):
        """Test EFOEmbedding model creation."""
        efo = EFOEmbedding(
            id="EFO_0004340", label="blood pressure", vector=[0.1, 0.2]
        )

        assert efo.id == "EFO_0004340"
        assert efo.label == "blood pressure"
        assert efo.vector == [0.1, 0.2]

    def test_model_result_model(self):
        """Test ModelResult model creation."""
        result = ModelResult(
            id=1,
            model="gpt-4",
            pmid="12345",
            metadata={"exposures": ["trait1"]},
            results={"score": 0.8},
        )

        assert result.id == 1
        assert result.model == "gpt-4"
        assert result.pmid == "12345"
        assert result.metadata == {"exposures": ["trait1"]}
        assert result.results == {"score": 0.8}

    def test_query_combination_model(self):
        """Test QueryCombination model creation."""
        combination = QueryCombination(
            id=1, pmid="12345", model="gpt-4", title="Test Study", trait_count=5
        )

        assert combination.id == 1
        assert combination.pmid == "12345"
        assert combination.model == "gpt-4"
        assert combination.title == "Test Study"
        assert combination.trait_count == 5


@pytest.mark.asyncio
class TestDatabaseHealthChecker:
    """Test database health checker functionality."""

    @pytest.fixture
    def mock_connections(self):
        """Create mock database connections."""
        vector_conn = Mock()
        trait_conn = Mock()

        # Mock basic health check queries
        vector_conn.execute.return_value.fetchone.return_value = [1]
        trait_conn.execute.return_value.fetchone.return_value = [1]

        return vector_conn, trait_conn

    async def test_health_check_success(self, mock_connections):
        """Test successful health check."""
        vector_conn, trait_conn = mock_connections

        health_checker = DatabaseHealthChecker(vector_conn, trait_conn)

        with patch.object(
            health_checker,
            "_run_performance_tests",
            return_value={"query_time": 0.1},
        ):
            status = await health_checker.perform_health_check()

            assert "overall_status" in status
            assert "vector_store" in status
            assert "trait_profile" in status
            assert "performance_metrics" in status

    async def test_health_check_with_error(self, mock_connections):
        """Test health check with database error."""
        vector_conn, trait_conn = mock_connections

        # Mock database error
        vector_conn.execute.side_effect = Exception("Database error")

        health_checker = DatabaseHealthChecker(vector_conn, trait_conn)

        status = await health_checker.perform_health_check()

        assert status["overall_status"] == "error"
        assert "error" in status


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components."""

    def test_end_to_end_trait_search(self):
        """Test end-to-end trait search scenario."""
        # Mock the full chain: repository -> service -> API response
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = [
            (1, "blood pressure", [0.1, 0.2]),
            (2, "diabetes", [0.3, 0.4]),
        ]
        mock_conn.execute.return_value.fetchone.return_value = (2,)  # Count

        trait_repo = TraitRepository(mock_conn)

        # Test search
        traits = trait_repo.search_traits("blood", limit=10, offset=0)
        count = trait_repo.get_count(
            "trait_embeddings", "trait_label ILIKE ?", ("%blood%",)
        )

        assert len(traits) == 2
        assert count == 2
        assert traits[0].trait_label == "blood pressure"

    def test_study_trait_relationship(self):
        """Test study-trait relationship queries."""
        mock_conn = Mock()

        # Mock study data
        mock_conn.execute.return_value.fetchone.return_value = (
            1,
            "gpt-4",
            "12345",
            {},
            {},
        )

        study_repo = StudyRepository(mock_conn)
        study = study_repo.get_study_by_id(1)

        assert study is not None
        assert study.pmid == "12345"

        # Mock trait data for the study
        mock_conn.execute.return_value.fetchall.return_value = [
            (1, 1, 5, "blood pressure", "bp1")
        ]

        traits = study_repo.get_study_traits(1)
        assert len(traits) == 1
        assert traits[0].trait_label == "blood pressure"


# Fixtures for test database setup
@pytest.fixture(scope="session")
def test_database_files():
    """Create temporary database files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_db_path = Path(temp_dir) / "test_vector_store.db"
        trait_db_path = Path(temp_dir) / "test_trait_profile.db"

        # Create minimal test databases
        vector_conn = duckdb.connect(str(vector_db_path))
        vector_conn.execute("""
            CREATE TABLE trait_embeddings (
                trait_index INTEGER PRIMARY KEY,
                trait_label VARCHAR NOT NULL,
                vector FLOAT[]
            )
        """)
        vector_conn.execute("""
            INSERT INTO trait_embeddings VALUES
            (1, 'blood pressure', [0.1, 0.2, 0.3]),
            (2, 'diabetes', [0.4, 0.5, 0.6])
        """)
        vector_conn.close()

        trait_conn = duckdb.connect(str(trait_db_path))
        trait_conn.execute("""
            CREATE TABLE query_combinations (
                id INTEGER PRIMARY KEY,
                pmid VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                trait_count INTEGER
            )
        """)
        trait_conn.execute("""
            INSERT INTO query_combinations VALUES
            (1, '12345', 'gpt-4', 'Test Study', 2)
        """)
        trait_conn.close()

        yield vector_db_path, trait_db_path


@pytest.mark.integration
class TestRealDatabaseIntegration:
    """Integration tests with real database connections."""

    @pytest.mark.asyncio
    async def test_connection_pool_with_real_files(self, test_database_files):
        """Test connection pool with real database files."""
        vector_db_path, trait_db_path = test_database_files

        config = DatabaseConfig(
            profile="test",
            vector_store_path=vector_db_path,
            trait_profile_path=trait_db_path,
            max_connections=2,
        )

        pool = DatabaseConnectionPool(config)
        await pool.initialize()

        # Test getting connections
        async with pool.get_vector_store_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM trait_embeddings"
            ).fetchone()
            assert result[0] == 2

        async with pool.get_trait_profile_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM query_combinations"
            ).fetchone()
            assert result[0] == 1

        await pool.close_all_connections()

    def test_trait_repository_real_queries(self, test_database_files):
        """Test trait repository with real database queries."""
        vector_db_path, _ = test_database_files

        conn = duckdb.connect(str(vector_db_path))
        repo = TraitRepository(conn)

        # Test getting trait by index
        trait = repo.get_trait_by_index(1)
        assert trait is not None
        assert trait.trait_label == "blood pressure"

        # Test searching traits
        traits = repo.search_traits("blood", limit=10, offset=0)
        assert len(traits) == 1
        assert traits[0].trait_label == "blood pressure"

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
