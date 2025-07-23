from fastapi import FastAPI

app = FastAPI(
    title="MR-KG Backend API",
    description="FastAPI backend for MR-KG project",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
