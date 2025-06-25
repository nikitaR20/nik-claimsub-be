from fastapi import FastAPI
from app.routes import claim, provider, segment

app = FastAPI()

@app.get("/hello")
def say_hello():
    return {"message": "Hello you"}

# Register API routers
app.include_router(claim.router)
app.include_router(provider.router)
app.include_router(segment.router)
