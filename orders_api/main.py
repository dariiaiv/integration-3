import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Response, status
from pydantic import BaseModel
import httpx
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orders-api")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:postgres_password_123@db:5432/orders_db")
FORECAST_API_URL = os.getenv("FORECAST_API_URL", "http://forecast-api:8001/predict")
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "50"))
JWT_SECRET = "supersecretjwtkey_12345"

http_client: httpx.AsyncClient = None
db_pool: asyncpg.Pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_pool
    http_client = httpx.AsyncClient(timeout=5.0)
    
    # Цикл очікування підключення до БД (до 15 спроб)
    for attempt in range(1, 16):
        try:
            logger.info(f"Спроба підключення до БД ({attempt}/15)...")
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            logger.info("Успішно підключено до PostgreSQL!")
            break
        except Exception as e:
            logger.warning(f"БД ще не готова ({e}), повтор через 2 сек...")
            await asyncio.sleep(2)
    else:
        raise RuntimeError("Не вдалося підключитися до БД")
    
    # Створення таблиць
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                client_name VARCHAR(100) NOT NULL,
                client_phone VARCHAR(50) NOT NULL,
                area FLOAT NOT NULL,
                rooms INT NOT NULL,
                predicted_price FLOAT,
                status VARCHAR(16) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
    yield
    await http_client.aclose()
    if db_pool:
        await db_pool.close()

app = FastAPI(title="Orders API", lifespan=lifespan)

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    if token != JWT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")

class OrderCreate(BaseModel):
    client_name: str
    client_phone: str
    area: float
    rooms: int

class OrderUpdate(BaseModel):
    client_name: Optional[str] = None
    client_phone: Optional[str] = None
    area: Optional[float] = None
    rooms: Optional[float] = None

async def calculate_forecast_bg(order_id: int, area: float, rooms: int):
    logger.info(f"[{datetime.now(timezone.utc)}] Фоновий розрахунок для замовлення #{order_id}")
    try:
        async with db_pool.acquire() as conn:
            exists = await conn.fetchval("SELECT id FROM orders WHERE id = $1", order_id)
            if not exists:
                return

        response = await http_client.post(FORECAST_API_URL, json={"area": area, "rooms": int(rooms)})
        if response.status_code == 200:
            price = response.json().get("predicted_price")
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE orders SET status = 'priced', predicted_price = $1 WHERE id = $2", price, order_id)
            logger.info(f"Замовлення #{order_id} розраховано: {price}")
        else:
            raise Exception("Forecast error")
    except Exception as e:
        logger.error(f"Помилка прогнозу для #{order_id}: {e}")
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE orders SET status = 'failed' WHERE id = $1", order_id)

@app.post("/orders", status_code=status.HTTP_202_ACCEPTED)
async def create_order(order: OrderCreate, background_tasks: BackgroundTasks, response: Response, auth: str = Header(None, alias="Authorization")):
    verify_token(auth)
    async with db_pool.acquire() as conn:
        order_id = await conn.fetchval(
            "INSERT INTO orders (client_name, client_phone, area, rooms, status) VALUES ($1, $2, $3, $4, 'pending') RETURNING id",
            order.client_name, order.client_phone, order.area, order.rooms
        )
    
    response.headers["Location"] = f"/orders/{order_id}"
    background_tasks.add_task(calculate_forecast_bg, order_id, order.area, order.rooms)
    return {"id": order_id, "status": "pending", "message": "Order created, forecast in progress"}

@app.post("/orders/batch")
async def create_orders_batch(orders: List[OrderCreate], auth: str = Header(None, alias="Authorization")):
    verify_token(auth)
    if len(orders) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=422, detail=f"Перевищено ліміт {BATCH_MAX_ITEMS}")

    async def process_single(index: int, item: OrderCreate):
        async with db_pool.acquire() as conn:
            order_id = await conn.fetchval(
                "INSERT INTO orders (client_name, client_phone, area, rooms, status) VALUES ($1, $2, $3, $4, 'pending') RETURNING id",
                item.client_name, item.client_phone, item.area, item.rooms
            )
        try:
            res = await http_client.post(FORECAST_API_URL, json={"area": item.area, "rooms": item.rooms}, timeout=2.0)
            if res.status_code == 200:
                price = res.json().get("predicted_price")
                async with db_pool.acquire() as conn:
                    await conn.execute("UPDATE orders SET status = 'priced', predicted_price = $1 WHERE id = $2", price, order_id)
                return {"index": index, "order_id": order_id, "status": "priced", "predicted_price": price}
            else:
                raise Exception("ForecastUnavailable")
        except Exception:
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE orders SET status = 'failed' WHERE id = $1", order_id)
            return {"index": index, "order_id": order_id, "status": "failed", "error": "ForecastUnavailable"}

    results = await asyncio.gather(*(process_single(i, item) for i, item in enumerate(orders)))
    return {"total": len(orders), "results": results}

@app.patch("/orders/{order_id}")
async def update_order(order_id: int, update: OrderUpdate, background_tasks: BackgroundTasks, auth: str = Header(None, alias="Authorization")):
    verify_token(auth)
    async with db_pool.acquire() as conn:
        curr = await conn.fetchrow("SELECT * FROM orders WHERE id = $1", order_id)
        if not curr:
            raise HTTPException(status_code=404, detail="Order not found")

        new_name = update.client_name if update.client_name is not None else curr["client_name"]
        new_phone = update.client_phone if update.client_phone is not None else curr["client_phone"]
        
        recalc = False
        new_area = curr["area"]
        new_rooms = curr["rooms"]
        if update.area is not None and update.area != curr["area"]:
            new_area = update.area
            recalc = True
        if update.rooms is not None and update.rooms != curr["rooms"]:
            new_rooms = int(update.rooms)
            recalc = True

        if recalc:
            await conn.execute(
                "UPDATE orders SET client_name=$1, client_phone=$2, area=$3, rooms=$4, status='pending', predicted_price=NULL WHERE id=$5",
                new_name, new_phone, new_area, new_rooms, order_id
            )
            background_tasks.add_task(calculate_forecast_bg, order_id, new_area, new_rooms)
            return {"id": order_id, "status": "pending", "message": "Property updated, recalculating forecast"}
        else:
            await conn.execute("UPDATE orders SET client_name=$1, client_phone=$2 WHERE id=$3", new_name, new_phone, order_id)
            return {"id": order_id, "status": curr["status"], "message": "Client details updated"}

@app.get("/orders")
async def get_orders(auth: str = Header(None, alias="Authorization")):
    verify_token(auth)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, client_name, area, rooms, status, predicted_price FROM orders ORDER BY id DESC")
        return [dict(r) for r in rows]

@app.get("/orders/{order_id}")
async def get_order(order_id: int, auth: str = Header(None, alias="Authorization")):
    verify_token(auth)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, client_name, area, rooms, status, predicted_price FROM orders WHERE id = $1", order_id)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        return dict(row)