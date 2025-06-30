from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = "mongodb+srv://fahim:Abcd1234@cluster0.ygq5j.mongodb.net/Gourmetique?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URI)
db = client["Gourmetique"]
collection = db["busytimes"]

# Cache to store trained models
trained_models = {}

# Request model for prediction
class PredictionRequest(BaseModel):
    hotelId: str
    timeSlot: str  # Format: "HH:MM"

# Function to train model for a hotelId
async def train_model_for_hotel(hotel_id: str):
    inputs = []
    outputs = []

    async for record in collection.find({"hotelId": hotel_id}):
        try:
            time_slot = int(record["timeSlot"].split(":")[0])  # extract hour
            number_of_customers = record["numberOfCustomers"]
            inputs.append([time_slot])
            outputs.append(number_of_customers)
        except Exception as e:
            logger.warning(f"Skipping bad record for {hotel_id}: {e}")

    if not inputs:
        logger.error(f"No data to train for hotelId {hotel_id}")
        return None

    model = LinearRegression()
    model.fit(np.array(inputs), np.array(outputs))
    return model

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    hotel_id = request.hotelId
    time_slot = request.timeSlot

    try:
        # Retrain or reuse model
        if hotel_id not in trained_models:
            model = await train_model_for_hotel(hotel_id)
            if model:
                trained_models[hotel_id] = model
            else:
                raise HTTPException(status_code=404, detail="No data available for this hotelId.")

        hour = int(time_slot.split(":")[0])
        prediction = trained_models[hotel_id].predict(np.array([[hour]]))

        return {
            "hotelId": hotel_id,
            "predictedNumberOfCustomers": round(prediction[0])
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail="Prediction error.")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI prediction service is running!"}

# Background task to monitor MongoDB changes
async def watch_for_changes():
    try:
        async for change in collection.watch():
            operation = change.get("operationType")

            if operation == "insert":
                hotel_id = change["fullDocument"]["hotelId"]
                time_slot = change["fullDocument"]["timeSlot"]
                try:
                    prediction_request = PredictionRequest(hotelId=hotel_id, timeSlot=time_slot)
                    prediction = await predict(prediction_request)
                    logger.info(f"Auto prediction (insert) for {hotel_id}: {prediction}")
                except Exception as e:
                    logger.error(f"Auto prediction failed on insert for {hotel_id}: {e}")

            elif operation == "delete":
                trained_models.clear()
                logger.info("Model cache cleared due to delete operation.")

            elif operation == "update":
                document_id = change["documentKey"]["_id"]
                updated_doc = await collection.find_one({"_id": document_id})
                if updated_doc:
                    hotel_id = updated_doc.get("hotelId")
                    time_slot = updated_doc.get("timeSlot", "12:00")  # default hour
                    try:
                        if hotel_id in trained_models:
                            del trained_models[hotel_id]  # retrain next time
                        prediction_request = PredictionRequest(hotelId=hotel_id, timeSlot=time_slot)
                        prediction = await predict(prediction_request)
                        logger.info(f"Auto prediction (update) for {hotel_id}: {prediction}")
                    except Exception as e:
                        logger.error(f"Auto prediction failed on update for {hotel_id}: {e}")
    except Exception as e:
        logger.error(f"Change stream error: {e}")

# Run the watcher in the background on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(watch_for_changes())
