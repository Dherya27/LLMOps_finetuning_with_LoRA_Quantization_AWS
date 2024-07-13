import os
import sys
from fastapi import FastAPI
import uvicorn
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from src.api.schemas import TrainingRequest
from src.api.schemas import TrainingConfig


# Place holder text and api object
app = FastAPI()

@app.post("/finetune")
async def predict_route(request: TrainingRequest):
    try:
        request_data = request
        if not request_data.config:
            request_data.config = TrainingConfig()

    except Exception as e:
        raise e

if __name__ == "__main__":
    import app
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

