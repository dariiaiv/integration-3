import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Forecast API")

class PropertyData(BaseModel):
    area: float
    rooms: int

@app.post("/predict")
async def predict_price(prop: PropertyData):
    await asyncio.sleep(0.3)
    base_price = prop.area * 1500.0 + prop.rooms * 5000.0
    return {"predicted_price": round(base_price, 2)}
