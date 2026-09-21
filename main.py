“””
POTENT ID Verification Microservice — free, self-hosted.
Two endpoints: OCR extraction from an ID photo, and face match between
a selfie and the ID photo. Both run on open-source libraries, no paid API.
“””
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io

app = FastAPI(title=“POTENT ID Verification Service”)

# Allow requests from your Netlify site specifically — tighten this to your

# real domain once deployed, instead of leaving it open to everyone.

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # TODO: replace with [“https://potentoperations.netlify.app”] once live
allow_methods=[“POST”],
allow_headers=[”*”],
)

_ocr = None
def get_ocr():
“”“Lazy-load PaddleOCR — only loads into memory on first real request,
so the free-tier server starts up fast instead of loading this at boot.”””
global _ocr
if _ocr is None:
from paddleocr import PaddleOCR
_ocr = PaddleOCR(use_angle_cls=True, lang=‘en’, show_log=False)
return _ocr

def read_image(file_bytes):
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
Real OCR, not a mock — but you (or the reviewing dispatcher) should still
glance at the photo yourself before trusting it for anything important.”””
contents = await file.read()
img = read_image(contents)
ocr = get_ocr()
result = ocr.ocr(img, cls=True)
lines = []
if result and result[0]:
for line in result[0]:
lines.append(line[1][0])  # the recognized text string
return {“raw_text_lines”: lines, “full_text”: “ | “.join(lines)}

@app.post(”/match-face”)
async def match_face(selfie: UploadFile = File(…), id_photo: UploadFile = File(…)):
“”“Compares a live selfie against the photo on the ID. Returns a
verified boolean and a distance score — lower distance = better match.
Real DeepFace comparison, no external API, no data leaves this server.”””
from deepface import DeepFace
selfie_bytes = await selfie.read()
id_bytes = await id_photo.read()
selfie_img = read_image(selfie_bytes)
id_img = read_image(id_bytes)
try:
result = DeepFace.verify(selfie_img, id_img, model_name=“VGG-Face”, enforce_detection=True)
except ValueError:
# enforce_detection raises this when it can’t find a clear face in one of the photos
raise HTTPException(status_code=400, detail=“Couldn’t clearly find a face in one of the photos. Try better lighting, facing the camera directly.”)
return {
“verified”: bool(result[“verified”]),
“distance”: float(result[“distance”]),
“threshold”: float(result[“threshold”]),
“confidence_note”: “Lower distance is a better match. If verified=true, the two photos are very likely the same person.”
}
