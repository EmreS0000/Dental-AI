"""
Dental AI v2 - FastAPI Backend (6 Sınıf + Mask Segmentasyon)
=============================================================
YOLO segmentasyon modeli ile dental röntgen analizi API.
"""
import os
import uuid
import logging
import base64
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np
import cv2
import io

from config import (
    UPLOAD_DIR, BASE_DIR, DISEASE_MODEL_PATH,
    DISEASE_CLASSES_V2, DISEASE_NAMES_TR, DISEASE_COLORS_HEX,
    TREATMENT_RECOMMENDATIONS_V2, FDI_NUMBERING
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dental-ai-v2")

# ═══════════════════════════════════════════════════════════════
# FastAPI Uygulama
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="🦷 Dental AI v2",
    description="6-sınıf dental hastalık tespiti + mask segmentasyon + FDI diş haritası + tedavi önerisi",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend statik dosyalar
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# ═══════════════════════════════════════════════════════════════
# Global Model Instance
# ═══════════════════════════════════════════════════════════════
_inference_engine = None

def get_engine():
    """Singleton inference engine"""
    global _inference_engine
    if _inference_engine is None:
        from inference_v2 import DentalInferenceV2
        _inference_engine = DentalInferenceV2()
    return _inference_engine


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Frontend index.html sun"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dental AI v2 API çalışıyor. Frontend bulunamadı."}


@app.get("/api/health")
async def health_check():
    """Sağlık kontrolü"""
    model_ok = DISEASE_MODEL_PATH.exists()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model_ok else "model_not_found",
        "model_path": str(DISEASE_MODEL_PATH),
        "classes": DISEASE_CLASSES_V2,
    }


@app.get("/api/v2/classes")
async def get_classes():
    """Hastalık sınıflarını ve renklerini döndür"""
    classes = []
    for cls in DISEASE_CLASSES_V2:
        classes.append({
            "id": DISEASE_CLASSES_V2.index(cls),
            "name": cls,
            "name_tr": DISEASE_NAMES_TR.get(cls, cls),
            "color": DISEASE_COLORS_HEX.get(cls, "#FFFFFF"),
            "treatment": TREATMENT_RECOMMENDATIONS_V2.get(cls, {}),
        })
    return {"classes": classes}


@app.get("/api/v2/tooth-map")
async def get_tooth_map():
    """FDI diş haritası bilgilerini döndür"""
    return {"fdi_numbering": FDI_NUMBERING}


@app.post("/api/v2/analyze")
async def analyze_xray_v2(
    xray_image: UploadFile = File(..., description="Dental röntgen görseli (JPEG/PNG)"),
    symptoms: str = Form(default="", description="Hasta semptomları"),
):
    """
    Ana analiz endpoint'i.
    - Dental röntgen görselini alır
    - YOLO ile hastalık tespiti yapar
    - Mask segmentasyondan diş konturu çıkarır
    - FDI numaralama + tedavi önerisi döndürür
    """
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]
    if xray_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz dosya tipi: {xray_image.content_type}. İzin verilen: {allowed_types}"
        )

    try:
        contents = await xray_image.read()
        image = Image.open(io.BytesIO(contents))

        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Görsel çok küçük. Minimum 100x100 piksel.")

        logger.info(f"📥 Röntgen alındı: {xray_image.filename} ({image.size[0]}x{image.size[1]})")

        # Görseli kaydet
        file_id = str(uuid.uuid4())[:8]
        save_filename = f"{file_id}_{xray_image.filename}"
        save_path = UPLOAD_DIR / save_filename
        with open(str(save_path), "wb") as f:
            f.write(contents)

        # Inference çalıştır
        engine = get_engine()
        enhanced_img, detections = engine.run_inference(str(save_path))

        # Görselleştirilmiş sonucu kaydet ve base64'e çevir
        vis_path = UPLOAD_DIR / f"vis_{save_filename}"
        vis_img = engine.visualize_results(enhanced_img, detections, str(vis_path))

        # Görselleştirilmiş imajı base64 olarak encode et
        _, buffer = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        vis_base64 = base64.b64encode(buffer).decode('utf-8')

        # Klinik rapor oluştur
        report = engine.generate_clinical_report(detections, xray_image.filename)

        logger.info(f"🔬 Sonuç: {report['summary']['total_findings']} bulgu, "
                    f"{report['summary']['affected_teeth']} etkilenen diş")

        response = {
            "success": True,
            "file_id": file_id,
            "image_info": {
                "width": image.size[0],
                "height": image.size[1],
                "filename": xray_image.filename,
            },
            "visualization": f"data:image/jpeg;base64,{vis_base64}",
            "report": report,
            "disease_colors": DISEASE_COLORS_HEX,
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analiz hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analiz başarısız: {str(e)}"
        )


@app.get("/api/v1/sample-images")
async def list_sample_images():
    """Demo için örnek röntgen görselleri listele"""
    from config import CHILDREN_CARIES_TRAIN, ADULT_TOOTH_TRAIN
    samples = []
    for dataset_dir in [CHILDREN_CARIES_TRAIN / "images", ADULT_TOOTH_TRAIN / "images"]:
        if dataset_dir.exists():
            for img_file in sorted(dataset_dir.glob("*.png"))[:5]:
                samples.append({
                    "filename": img_file.name,
                    "path": str(img_file),
                    "dataset": dataset_dir.parent.parent.name,
                })
    return {"samples": samples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
