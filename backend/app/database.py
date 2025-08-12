"""Database management for the FastAPI backend."""

import duckdb
import pandas as pd
from typing import List, Optional

from app.config import Settings


class DatabaseManager:
    """Manages database connections and queries."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.trait_conn: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
    
    def _connect(self):
        """Establish database connections."""
        try:
            self.vector_conn = duckdb.connect(
                self.settings.vector_store_db_path, 
                read_only=True
            )
            self.trait_conn = duckdb.connect(
                self.settings.trait_profile_db_path, 
                read_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to databases: {e}")
    
    def close(self):
        """Close database connections."""
        if self.vector_conn:
            self.vector_conn.close()
        if self.trait_conn:
            self.trait_conn.close()
    
    def get_top_traits(self, limit: int = 50) -> pd.DataFrame:
        """Get top trait labels by appearance count."""
        query = """
        SELECT DISTINCT trait_label, COUNT(*) as appearance_count
        FROM model_result_traits
        GROUP BY trait_label
        ORDER BY appearance_count DESC
        LIMIT ?
        """
        result = self.vector_conn.execute(query, [limit]).fetchdf()
        return result
    
    def search_traits(self, filter_text: str, limit: int = 100) -> pd.DataFrame:
        """Search for traits matching the filter text."""
        if not filter_text:
            return self.get_top_traits(limit)
        
        query = """
        SELECT DISTINCT trait_label, COUNT(*) as appearance_count
        FROM model_result_traits
        WHERE trait_label ILIKE ?
        GROUP BY trait_label
        ORDER BY appearance_count DESC
        LIMIT ?
        """
        search_pattern = f"%{filter_text}%"
        result = self.vector_conn.execute(query, [search_pattern, limit]).fetchdf()
        return result
    
    def get_total_traits_count(self) -> int:
        """Get total number of unique traits."""
        query = "SELECT COUNT(DISTINCT trait_label) FROM model_result_traits"
        result = self.vector_conn.execute(query).fetchone()
        return result[0] if result else 0
    
    def get_studies_for_trait_and_model(
        self, trait_label: str, model: str, limit: int = 100
    ) -> pd.DataFrame:
        """Get studies for a specific trait and model."""
        query = """
        SELECT DISTINCT
            mr.id as model_result_id,
            mr.pmid,
            pubmed.title,
            pubmed.journal,
            pubmed.pub_date,
            mr.metadata
        FROM model_results mr
        JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        LEFT JOIN mr_pubmed_data pubmed ON mr.pmid = pubmed.pmid
        WHERE mrt.trait_label = ? 
        AND mr.model = ?
        ORDER BY pubmed.pub_date DESC, mr.pmid
        LIMIT ?
        """
        result = self.vector_conn.execute(query, [trait_label, model, limit]).fetchdf()
        return result
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        query = "SELECT DISTINCT model FROM model_results ORDER BY model"
        result = self.vector_conn.execute(query).fetchall()
        return [row[0] for row in result]
