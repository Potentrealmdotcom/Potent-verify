“””
POTENT ID Verification Microservice — free, self-hosted.
Two endpoints: OCR extraction from an ID photo, and face match between
a selfie and the ID photo. Both run on lightweight, open-source
libraries chosen specifically to fit a free-tier 512MB server —
no TensorFlow, no heavyweight ML frameworks.
“””
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import face_recognition
import io
import numpy as np

app = FastAPI(title=“POTENT ID Verification Service”)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],  # TODO: replace with [“https://potentoperations.netlify.app”] once live
allow_methods=[“POST”],
allow_headers=[”*”],
)

@app.get(”/”)
def health():
return {“status”: “ok”, “service”: “POTENT ID Verification”}

@app.post(”/extract-id”)
async def extract_id(file: UploadFile = File(…)):
“”“Takes a photo of a driver’s license/ID, returns the raw text found on it.
Uses Tesseract OCR — real extraction, still worth a human glance at the
photo before trusting it for anything important.”””
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
“”“Compares a live selfie against the photo on the ID. Uses face_recognition
(built on dlib) — no TensorFlow, genuinely lightweight, real comparison,
no data leaves this server.”””
selfie_bytes = await selfie.read()
id_bytes = await id_photo.read()

```
selfie_img = face_recognition.load_image_file(io.BytesIO(selfie_bytes))
id_img = face_recognition.load_image_file(io.BytesIO(id_bytes))

selfie_encodings = face_recognition.face_encodings(selfie_img)
id_encodings = face_recognition.face_encodings(id_img)

if not selfie_encodings:
    raise HTTPException(status_code=400, detail="Couldn't find a clear face in the selfie. Try better lighting, facing the camera directly.")
if not id_encodings:
    raise HTTPException(status_code=400, detail="Couldn't find a clear face on the ID photo. Try a clearer, well-lit photo of the ID.")

face_distance = face_recognition.face_distance([id_encodings[0]], selfie_encodings[0])[0]
is_match = bool(face_recognition.compare_faces([id_encodings[0]], selfie_encodings[0], tolerance=0.6)[0])

return {
    "verified": is_match,
    "distance": float(face_distance),
    "threshold": 0.6,
    "confidence_note": "Lower distance is a better match (0 = identical). If verified=true, the two photos are very likely the same person."
}
```
