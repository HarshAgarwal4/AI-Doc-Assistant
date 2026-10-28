from fastapi import FastAPI , status , HTTPException

app = FastAPI(
    title="AI Doc Assistant",
    description="My first FastAPI backend",
    version="1.0.0"
)

@app.get("/" , status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to AI Doc Assistant"}


@app.get("/health" , status_code=status.HTTP_200_OK )
def health_check():
    return {"status": "ok"}

