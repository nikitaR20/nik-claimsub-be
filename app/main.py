from fastapi import FastAPI
from app import models, database
from app.routers import claim, provider, segment

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.include_router(provider.router)
app.include_router(segment.router)
app.include_router(claim.router)
