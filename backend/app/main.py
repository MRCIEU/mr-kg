"""FastAPI backend for MR-KG web application."""

from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.database import DatabaseManager


class TraitLabel(BaseModel):
    """Trait label with appearance count."""
    trait_label: str
    appearance_count: int


class TraitSearchResponse(BaseModel):
    """Response for trait search."""
    traits: List[TraitLabel]
    total_count: int
    filtered_count: int


class StudyResult(BaseModel):
    """Study result for a trait and model."""
    model_result_id: int
    pmid: str
    title: Optional[str]
    journal: Optional[str]
    pub_date: Optional[str]
    metadata: Optional[dict]


class StudyResultsResponse(BaseModel):
    """Response for study results."""
    studies: List[StudyResult]
    trait_label: str
    model: str
    total_count: int


# Global database manager
db_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global db_manager
    settings = get_settings()
    db_manager = DatabaseManager(settings)
    yield
    if db_manager:
        db_manager.close()


app = FastAPI(
    title="MR-KG Backend API",
    description="FastAPI backend for exploring MR-KG structural PubMed literature data",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vue dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "MR-KG Backend API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/traits/search", response_model=TraitSearchResponse)
async def search_traits(
    filter_text: str = Query("", description="Filter text for trait labels"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
):
    """Search for trait labels."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        traits_df = db_manager.search_traits(filter_text, limit)
        total_count = db_manager.get_total_traits_count()
        
        traits = [
            TraitLabel(trait_label=row["trait_label"], appearance_count=row["appearance_count"])
            for _, row in traits_df.iterrows()
        ]
        
        return TraitSearchResponse(
            traits=traits,
            total_count=total_count,
            filtered_count=len(traits),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/traits/top", response_model=TraitSearchResponse)
async def get_top_traits(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
):
    """Get top trait labels by appearance count."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        traits_df = db_manager.get_top_traits(limit)
        total_count = db_manager.get_total_traits_count()
        
        traits = [
            TraitLabel(trait_label=row["trait_label"], appearance_count=row["appearance_count"])
            for _, row in traits_df.iterrows()
        ]
        
        return TraitSearchResponse(
            traits=traits,
            total_count=total_count,
            filtered_count=len(traits),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/traits/{trait_label}/studies", response_model=StudyResultsResponse)
async def get_studies_for_trait(
    trait_label: str,
    model: str = Query(..., description="Model name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
):
    """Get studies for a specific trait and model."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        studies_df = db_manager.get_studies_for_trait_and_model(trait_label, model, limit)
        
        studies = [
            StudyResult(
                model_result_id=row["model_result_id"],
                pmid=row["pmid"],
                title=row.get("title"),
                journal=row.get("journal"),
                pub_date=row.get("pub_date"),
                metadata=row.get("metadata"),
            )
            for _, row in studies_df.iterrows()
        ]
        
        return StudyResultsResponse(
            studies=studies,
            trait_label=trait_label,
            model=model,
            total_count=len(studies),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """Get list of available models."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        models = db_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
