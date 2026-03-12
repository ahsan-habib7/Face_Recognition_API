# ============================================================
# Face Recognition & ID Validation API
# Built with FastAPI + InsightFace (ArcFace) + Tesseract OCR
# ============================================================
# Two endpoints:
#   POST /validate-id    → OCR checks if image is a Bangladesh ID
#   POST /verify-face    → ArcFace compares ID photo vs selfie
#   GET  /health         → liveness probe
# ============================================================

import cv2
import numpy as np
import pytesseract
from insightface.app import FaceAnalysis

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Bank Face Verification API",
    description="ID validation + ArcFace face verification for Bangladesh bank KYC.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load InsightFace ArcFace model once at startup ─────────
print("Loading InsightFace ArcFace model...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model loaded successfully.")


# ══════════════════════════════════════════════════════════════
# KEYWORD PATTERNS — Bangladesh ID documents
# These text strings ONLY appear on official Bangladesh IDs.
# ══════════════════════════════════════════════════════════════

NID_KEYWORDS = [
    "জাতীয় পরিচয়পত্র",
    "national id card",
    "election commission",
    "nid no",
    "voter no",
    "bangladesh election",
    "নির্বাচন কমিশন",
    "জন্ম তারিখ",
    "রক্তের গ্রুপ",
    "id no",
]

PASSPORT_KEYWORDS = [
    "p<bgd",
    "republic of bangladesh",
    "গণপ্রজাতন্ত্রী বাংলাদেশ",
    "passport no",
    "bmet",
    "bangladesh passport",
    "place of birth",
    "date of expiry",
    "nationality",
]

DRIVING_LICENSE_KEYWORDS = [
    "brta",
    "bangladesh road transport authority",
    "বাংলাদেশ সড়ক পরিবহন কর্তৃপক্ষ",
    "driving licence",
    "driving license",
    "licence no",
    "license no",
    "vehicle class",
    "dl no",
]


# ── Helper: decode image bytes → NumPy BGR ─────────────────
def decode_image(file_bytes: bytes, label: str) -> np.ndarray:
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode {label}. Please upload a valid JPEG or PNG."
        )
    return image


# ── Helper: preprocess image for better OCR accuracy ───────
def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    # Upscale small images — Tesseract is more accurate on larger images
    if w < 1000:
        scale = 1000 / w
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


# ── Helper: run Tesseract OCR ──────────────────────────────
def extract_text(image_bgr: np.ndarray) -> str:
    processed = preprocess_for_ocr(image_bgr)
    config    = r'--oem 3 --psm 3'
    try:
        # Try English + Bengali together
        text = pytesseract.image_to_string(processed, lang='eng+ben', config=config)
    except Exception:
        # Fallback to English only
        text = pytesseract.image_to_string(processed, lang='eng', config=config)
    return text.lower().strip()


# ── Helper: match OCR text against ID keyword lists ────────
def detect_id_type(text: str) -> dict:
    text_lower = text.lower()

    nid_matches = [kw for kw in NID_KEYWORDS if kw.lower() in text_lower]
    if nid_matches:
        return {
            "valid":   True,
            "id_type": "Bangladesh NID",
            "message": "Valid Bangladesh National ID Card detected"
        }

    passport_matches = [kw for kw in PASSPORT_KEYWORDS if kw.lower() in text_lower]
    if passport_matches:
        return {
            "valid":   True,
            "id_type": "Bangladesh Passport",
            "message": "Valid Bangladesh Passport detected"
        }

    dl_matches = [kw for kw in DRIVING_LICENSE_KEYWORDS if kw.lower() in text_lower]
    if dl_matches:
        return {
            "valid":   True,
            "id_type": "Bangladesh Driving License",
            "message": "Valid Bangladesh Driving License detected"
        }

    return {
        "valid":   False,
        "id_type": "unknown",
        "message": "No valid Bangladesh ID detected. Please upload your NID, Passport, or Driving License."
    }


# ══════════════════════════════════════════════════════════════
# ENDPOINT 1: POST /validate-id
# Step 1 — OCR validates the uploaded image is a Bangladesh ID
# ══════════════════════════════════════════════════════════════
@app.post("/validate-id")
async def validate_id(
    id_image: UploadFile = File(..., description="ID card image to validate")
):
    """
    Validates whether the uploaded image is a genuine Bangladesh ID.

    Returns:
    ```json
    {"valid": true, "id_type": "Bangladesh NID", "message": "..."}
    ```
    Or raises HTTP 400 if not a valid ID.
    """
    image_bytes    = await id_image.read()
    image_bgr      = decode_image(image_bytes, "ID image")
    extracted_text = extract_text(image_bgr)
    result         = detect_id_type(extracted_text)

    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return {
        "valid":   result["valid"],
        "id_type": result["id_type"],
        "message": result["message"],
    }


# ══════════════════════════════════════════════════════════════
# ENDPOINT 2: POST /verify-face
# Step 2 — ArcFace comparison of ID card face vs live selfie
# ══════════════════════════════════════════════════════════════

def get_face_embedding(image_bgr: np.ndarray, label: str) -> np.ndarray:
    faces = face_app.get(image_bgr)
    if not faces:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected in the {label}. Please upload a clear photo."
        )
    return max(faces, key=lambda f: f.det_score).embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


@app.post("/verify-face")
async def verify_face(
    id_image:     UploadFile = File(..., description="ID card photo"),
    selfie_image: UploadFile = File(..., description="Live selfie"),
):
    id_bytes     = await id_image.read()
    selfie_bytes = await selfie_image.read()

    id_img     = decode_image(id_bytes,     "ID image")
    selfie_img = decode_image(selfie_bytes, "selfie image")

    id_embedding     = get_face_embedding(id_img,     "ID image")
    selfie_embedding = get_face_embedding(selfie_img, "selfie image")

    similarity = cosine_similarity(id_embedding, selfie_embedding)
    threshold  = 0.5
    matched    = similarity >= threshold

    if   similarity >= 0.7: confidence = "Very High"
    elif similarity >= 0.6: confidence = "High"
    elif similarity >= 0.5: confidence = "Medium"
    elif similarity >= 0.3: confidence = "Low"
    else:                   confidence = "Very Low"

    return {
        "match":      matched,
        "message":    "Faces match" if matched else "Faces do not match",
        "similarity": round(similarity, 4),
        "confidence": confidence,
        "threshold":  threshold,
    }


# ── Health check ───────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":    "ok",
        "service":   "bank-face-verification-api",
        "model":     "InsightFace ArcFace buffalo_l",
        "endpoints": ["/validate-id", "/verify-face", "/health"]
    }
