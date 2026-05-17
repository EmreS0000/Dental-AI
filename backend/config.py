"""
Dental AI v2 - Konfigürasyon (6 Sınıf)
========================================
"""
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Dizin yapısı
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_ROOT = PROJECT_ROOT

# Dataset yolları
CHILDREN_CARIES_TRAIN = DATASET_ROOT / "Childrens dental caries segmentation dataset" / "Train"
CHILDREN_CARIES_TEST = DATASET_ROOT / "Childrens dental caries segmentation dataset" / "Test"
CHILDREN_CARIES_SUPPLEMENTAL = DATASET_ROOT / "Childrens dental caries segmentation dataset" / "Supplemental content93"
ADULT_TOOTH_TRAIN = DATASET_ROOT / "Adult tooth segmentation dataset" / "Dataset and code" / "train"
ADULT_TOOTH_TEST = DATASET_ROOT / "Adult tooth segmentation dataset" / "Dataset and code" / "test"
PANORAMIC_DATA = DATASET_ROOT / "Adult tooth segmentation dataset" / "Panoramic radiography database"
ARCHIVE_DATA = DATASET_ROOT / "Adult tooth segmentation dataset" / "Archive"
PEDIATRIC_TRAIN = DATASET_ROOT / "Pediatric dental disease detection dataset" / "Train"
PEDIATRIC_TEST = DATASET_ROOT / "Pediatric dental disease detection dataset" / "Test"
DENTEX_DATA = DATASET_ROOT / "Dentex" / "training_data" / "training_data"
TUFTS_DATA = DATASET_ROOT / "tufts dental" / "Expert" / "Expert"

# Model dizinleri
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# v2: YOLO Model Yolu (6 Sınıf)
# ═══════════════════════════════════════════════════════════════
YOLO_V2_ROOT = Path("C:/Users/emres/DentalAI_YOLO_v2")
YOLO_V2_MODEL_PATH = YOLO_V2_ROOT / "runs" / "dental_v2_6class" / "weights" / "best.pt"
YOLO_V2_YAML = YOLO_V2_ROOT / "dataset.yaml"

# ═══════════════════════════════════════════════════════════════
# v2: 6 Sınıf (verisi olmayan 4 sınıf kaldırıldı)
# ═══════════════════════════════════════════════════════════════
DISEASE_CLASSES_V2 = [
    "caries",                  # 0 - Çürük
    "deep_caries",             # 1 - Derin Çürük
    "periapical_lesion",       # 2 - Periapikal Lezyon
    "impacted_tooth",          # 3 - Gömülü Diş
    "non_odontogenic_lesion",  # 4 - Non-odontogenic Lezyon
    "pericoronal_lesion",      # 5 - Perikoronal Lezyon
]

# Hastalık Türkçe isimleri
DISEASE_NAMES_TR = {
    "caries": "Çürük",
    "deep_caries": "Derin Çürük",
    "periapical_lesion": "Periapikal Lezyon",
    "impacted_tooth": "Gömülü Diş",
    "non_odontogenic_lesion": "Non-odontogenic Lezyon",
    "pericoronal_lesion": "Perikoronal Lezyon",
}

# Hastalık renkleri (BGR formatında, görselleştirme için)
DISEASE_COLORS = {
    "caries":                  (0, 165, 255),   # Turuncu
    "deep_caries":             (0, 0, 255),      # Kırmızı
    "periapical_lesion":       (255, 0, 255),    # Magenta
    "impacted_tooth":          (255, 255, 0),    # Camgöbeği
    "non_odontogenic_lesion":  (0, 255, 0),      # Yeşil
    "pericoronal_lesion":      (255, 0, 0),      # Mavi
}

# Hastalık renkleri (HEX - Frontend için)
DISEASE_COLORS_HEX = {
    "caries":                  "#FFA500",
    "deep_caries":             "#FF0000",
    "periapical_lesion":       "#FF00FF",
    "impacted_tooth":          "#00FFFF",
    "non_odontogenic_lesion":  "#00FF00",
    "pericoronal_lesion":      "#0000FF",
}

# ═══════════════════════════════════════════════════════════════
# FDI Diş Numaralama Sistemi
# ═══════════════════════════════════════════════════════════════
FDI_NUMBERING = {
    # Sağ Üst (Quadrant 1)
    "11": "Sağ Üst Santral Kesici",
    "12": "Sağ Üst Lateral Kesici",
    "13": "Sağ Üst Kanin",
    "14": "Sağ Üst 1. Küçük Azı",
    "15": "Sağ Üst 2. Küçük Azı",
    "16": "Sağ Üst 1. Büyük Azı",
    "17": "Sağ Üst 2. Büyük Azı",
    "18": "Sağ Üst 3. Büyük Azı (20'lik)",
    # Sol Üst (Quadrant 2)
    "21": "Sol Üst Santral Kesici",
    "22": "Sol Üst Lateral Kesici",
    "23": "Sol Üst Kanin",
    "24": "Sol Üst 1. Küçük Azı",
    "25": "Sol Üst 2. Küçük Azı",
    "26": "Sol Üst 1. Büyük Azı",
    "27": "Sol Üst 2. Büyük Azı",
    "28": "Sol Üst 3. Büyük Azı (20'lik)",
    # Sol Alt (Quadrant 3)
    "31": "Sol Alt Santral Kesici",
    "32": "Sol Alt Lateral Kesici",
    "33": "Sol Alt Kanin",
    "34": "Sol Alt 1. Küçük Azı",
    "35": "Sol Alt 2. Küçük Azı",
    "36": "Sol Alt 1. Büyük Azı",
    "37": "Sol Alt 2. Büyük Azı",
    "38": "Sol Alt 3. Büyük Azı (20'lik)",
    # Sağ Alt (Quadrant 4)
    "41": "Sağ Alt Santral Kesici",
    "42": "Sağ Alt Lateral Kesici",
    "43": "Sağ Alt Kanin",
    "44": "Sağ Alt 1. Küçük Azı",
    "45": "Sağ Alt 2. Küçük Azı",
    "46": "Sağ Alt 1. Büyük Azı",
    "47": "Sağ Alt 2. Büyük Azı",
    "48": "Sağ Alt 3. Büyük Azı (20'lik)",
}

# ═══════════════════════════════════════════════════════════════
# Tedavi Önerileri (v2 - 6 sınıf)
# ═══════════════════════════════════════════════════════════════
TREATMENT_RECOMMENDATIONS_V2 = {
    "caries": {
        "tr": "Çürük temizliği ve kompozit/amalgam dolgu uygulaması önerilir. Çürük derin ise endodontik değerlendirme yapılmalıdır.",
        "en": "Caries removal and composite/amalgam filling is recommended. If deep, endodontic evaluation should be performed.",
        "priority": "orta",
    },
    "deep_caries": {
        "tr": "Derin çürük tespit edilmiştir. Acil kanal tedavisi (endodonti) gerekebilir. Pulpa durumu değerlendirilmelidir.",
        "en": "Deep caries detected. Urgent root canal treatment may be needed. Pulp status should be evaluated.",
        "priority": "yüksek",
    },
    "periapical_lesion": {
        "tr": "Kanal tedavisi (endodonti) önerilir. Lezyonun boyutuna göre apikal cerrahi değerlendirilebilir.",
        "en": "Root canal treatment (endodontics) is recommended. Apical surgery may be considered depending on lesion size.",
        "priority": "yüksek",
    },
    "impacted_tooth": {
        "tr": "Gömülü diş çevre dokulara etkisi açısından değerlendirilmelidir. Gerekirse cerrahi çekim planlanmalıdır.",
        "en": "Impacted tooth should be evaluated for effects on surrounding tissues. Surgical extraction may be planned if necessary.",
        "priority": "orta",
    },
    "non_odontogenic_lesion": {
        "tr": "Non-odontogenic lezyon tespit edilmiştir. İleri görüntüleme (CBCT/MR) ve biyopsi ile histopatolojik değerlendirme önerilir.",
        "en": "Non-odontogenic lesion detected. Advanced imaging (CBCT/MR) and histopathological evaluation via biopsy is recommended.",
        "priority": "yüksek",
    },
    "pericoronal_lesion": {
        "tr": "Perikoronal lezyon (genellikle gömülü dişle ilişkili). Cerrahi çekim ve küretaj önerilir.",
        "en": "Pericoronal lesion (usually related to impacted tooth). Surgical extraction and curettage is recommended.",
        "priority": "orta",
    },
}

# ═══════════════════════════════════════════════════════════════
# Inference Ayarları
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3
MAX_DETECTIONS = 100
INFERENCE_IMGSZ = 640

# Eski config uyumluluğu
TOOTH_CLASSES = ["__background__", "tooth"]
DISEASE_CLASSES = DISEASE_CLASSES_V2
TREATMENT_RECOMMENDATIONS = {k: {"tr": v["tr"], "en": v["en"]} for k, v in TREATMENT_RECOMMENDATIONS_V2.items()}
TOOTH_MODEL_PATH = MODEL_DIR / "tooth_segmentation_maskrcnn.pth"
DISEASE_MODEL_PATH = MODEL_DIR / "disease_detection_maskrcnn.pth"
TARGET_IMAGE_SIZE = (800, 800)
