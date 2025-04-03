# File: app.py
import asyncio
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Any

import webappdriver

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Real-time Data Stream API")

# Add CORS middleware to allow requests from the React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class to manage WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


# Data generator class
class DataGenerator:
    def __init__(self):
        self.base_value = 50.0
        self.noise_factor = 5.0
        self.trend_factor = 0.1
        self.trend_direction = 1  # 1 for up, -1 for down
        self.last_trend_change = time.time()

    def generate_data_point(self) -> Dict[str, Any]:
        # Change trend direction occasionally
        current_time = time.time()
        if current_time - self.last_trend_change > 10:  # Change every 10 seconds
            if random.random() < 0.3:  # 30% chance to change direction
                self.trend_direction *= -1
                self.last_trend_change = current_time

        # Apply trend
        self.base_value += self.trend_factor * self.trend_direction
        
        # Apply random noise
        noise = (random.random() - 0.5) * 2 * self.noise_factor
        value = self.base_value + noise
        
        # Keep value in a reasonable range
        if value < 0:
            value = 0
        if value > 100:
            value = 100
            self.trend_direction = -1  # Force trend down if we hit max
        return {"name": datetime.now().strftime("%H:%M:%S"), "value": round(value, 2), "timestamp": datetime.now().isoformat(), "is_safe": False, "amount_to_go_for_safe": 0}
        # return {
        #     "name": datetime.now().strftime("%H:%M:%S"),
        #     "value": round(value, 2),
        #     "timestamp": datetime.now().isoformat()
        # }


manager = ConnectionManager()
data_generator = DataGenerator()


@app.get("/")
async def get_root():
    return {"message": "Real-time Data Stream API. Connect to /ws for WebSocket stream."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial data point
        initial_data = data_generator.generate_data_point()
        await manager.send_personal_message(json.dumps(initial_data), websocket)
        
        # Set up data streaming task
        data_stream_task = asyncio.create_task(stream_data(websocket))
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "request_data":
                    # Send immediate data point on request
                    point = data_generator.generate_data_point()
                    await manager.send_personal_message(json.dumps(point), websocket)
            except json.JSONDecodeError:
                print(f"Received invalid JSON: {data}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def stream_data(websocket: WebSocket):
    """Task to periodically send data to the client"""
    # hier: laad van de pickle file de data in
    # en laad de classificatie van de hele data (van de definitie, niet model)
    # en laad het model in
    webappdriver.init()
    
    try:
        data_generator = webappdriver.get_data_point()
        while True:
            await asyncio.sleep(.5)  # Send data every second
            # hier kan je de timer aanpassen ook
            # data_point = {"value": round(10, 2), "is_safe": False, "amount_to_go_for_safe": 0}
            data_point = next(data_generator)
            
            # mogelijke extensie: heave_waarde -> PCA of meerdere grafieken van heave en rol en wat er nodig is

        

            await manager.send_personal_message(json.dumps(data_point), websocket)
    except Exception as e:
        print(f"Error in data stream: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)