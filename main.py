“””
POTENT ID Verification Microservice — free, self-hosted.
Two endpoints: OCR extraction from an ID photo, and face match between
a selfie and the ID photo. Both run on open-source libraries, no paid API.
“””
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io

app = FastAPI(title=“POTENT ID Verification Service”)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # TODO: replace with [“https://potentoperations.netlify.app”] once live
allow_methods=[“POST”],
allow_headers=[”*”],
)

def read_image_cv2(file_bytes):
arr = np.frombuffer(file_bytes, np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img is None:
raise HTTPException(status_code=400, detail=“Could not read that image. Try a clearer photo.”)
return img

@app.get(”/”)
def health():
return {“status”: “ok”, “service”: “POTENT ID Verification”}

@app.post(”/extract-id”)
async def extract_id(file: UploadFile = File(…)):
“”“Takes a photo of a driver’s license/ID, returns the raw text found on it.
Uses Tesseract OCR — real extraction, not a mock. Still worth a human
glance at the photo before trusting it for anything important.”””
contents = await file.read()
try:
img = Image.open(io.BytesIO(contents))
except Exception:
raise HTTPException(status_code=400, detail=“Could not read that image. Try a clearer photo.”)
text = pytesseract.image_to_string(img)
lines = [l.strip() for l in text.split(”\n”) if l.strip()]
return {“raw_text_lines”: lines, “full_text”: “ | “.join(lines)}

@app.post(”/match-face”)
async def match_face(selfie: UploadFile = File(…), id_photo: UploadFile = File(…)):
“”“Compares a live selfie against the photo on the ID. Returns a
verified boolean and a distance score — lower distance = better match.
Real DeepFace comparison, no external API, no data leaves this server.”””
from deepface import DeepFace
selfie_bytes = await selfie.read()
id_bytes = await id_photo.read()
selfie_img = read_image_cv2(selfie_bytes)
id_img = read_image_cv2(id_bytes)
try:
result = DeepFace.verify(selfie_img, id_img, model_name=“VGG-Face”, enforce_detection=True)
except ValueError:
raise HTTPException(status_code=400, detail=“Couldn’t clearly find a face in one of the photos. Try better lighting, facing the camera directly.”)
return {
“verified”: bool(result[“verified”]),
“distance”: float(result[“distance”]),
“threshold”: float(result[“threshold”]),
“confidence_note”: “Lower distance is a better match. If verified=true, the two photos are very likely the same person.”
}
