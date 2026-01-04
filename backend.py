from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
from typing import Dict, Any, Optional, List, Tuple
import os
from dotenv import load_dotenv
import logging
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = "Hacktrix-121/deberta-v3-base-ingredients"

app = FastAPI(
    title="Ingredient Risk Classifier API",
    description="API for classifying food ingredient risk levels",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# Global OCR instance (lazy initialization)
_ocr_instance = None

def get_ocr():
    """Lazy load EasyOCR only when needed."""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            import easyocr
            _ocr_instance = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            raise HTTPException(status_code=500, detail=f"OCR initialization failed: {str(e)}")
    return _ocr_instance

# Mapping dictionaries
id2risk_level = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
risk_level2category = {
    1: "Very Safe",
    2: "Safe",
    3: "Moderate",
    4: "Concerning",
    5: "High Risk",
}

# Request/Response models
class IngredientsRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    pred_id: int
    risk_level: int
    risk_category: str
    probabilities: Dict[str, float]
    error: Optional[str] = None

class OCRResult(BaseModel):
    text: str
    confidence: float

class ImagePredictionResponse(BaseModel):
    extracted_text: str
    ocr_results: List[OCRResult]
    pred_id: int
    risk_level: int
    risk_category: str
    probabilities: Dict[str, float]
    error: Optional[str] = None

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: IngredientsRequest) -> PredictionResponse:
    """Predict the risk level of ingredients."""
    text = payload.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty input text")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Make prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        # Process results
        pred_id = int(torch.argmax(probs).item())
        risk_level = id2risk_level.get(pred_id, 3)
        risk_category = risk_level2category.get(risk_level, "Unknown")
        
        # Format probabilities
        probabilities = {str(i): float(p) for i, p in enumerate(probs.tolist()[0])}
        
        logger.info(f"✅ Successfully classified ingredients with risk level {risk_level}")
        
        return PredictionResponse(
            text=text,
            pred_id=pred_id,
            risk_level=risk_level,
            risk_category=risk_category,
            probabilities=probabilities,
            error=None
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-image", response_model=ImagePredictionResponse)
def predict_risk_from_image(file: UploadFile = File(...)) -> ImagePredictionResponse:
    """Predict the risk level of ingredients extracted from an image using OCR."""
    
    # Get OCR instance (lazy loaded)
    ocr = get_ocr()
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")
    
    try:
        # Read image file
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Perform OCR
        ocr_result = ocr.readtext(image)
        
        if not ocr_result:
            raise HTTPException(status_code=400, detail="No text detected in the image")
        
        # Extract text and confidence scores
        ocr_details = []
        full_text = []
        
        for detection in ocr_result:
            text_segment = detection[1]  # EasyOCR format: (bbox, text, confidence)
            confidence = detection[2]
            full_text.append(text_segment)
            ocr_details.append({
                "text": text_segment,
                "confidence": float(confidence)
            })
        
        extracted_text = " ".join(full_text)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the image")
        
        # Classify the extracted text
        inputs = tokenizer(
            extracted_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        # Process results
        pred_id = int(torch.argmax(probs).item())
        risk_level = id2risk_level.get(pred_id, 3)
        risk_category = risk_level2category.get(risk_level, "Unknown")
        
        # Format probabilities
        probabilities = {str(i): float(p) for i, p in enumerate(probs.tolist()[0])}
        
        logger.info(f"✅ Successfully classified OCR text with risk level {risk_level}")
        
        return ImagePredictionResponse(
            extracted_text=extracted_text,
            ocr_results=[OCRResult(**detail) for detail in ocr_details],
            pred_id=pred_id,
            risk_level=risk_level,
            risk_category=risk_category,
            probabilities=probabilities,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Image prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Ingredient Risk Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
