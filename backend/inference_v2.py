"""
Dental AI v3 - Inference Pipeline (Mask R-CNN PyTorch)
======================================================
"""
import cv2
import torch
import numpy as np
import json
import torchvision
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F_trans

from med_preprocess import enhance_dental_image
from config import (
    DISEASE_MODEL_PATH, DISEASE_CLASSES_V2, DISEASE_NAMES_TR,
    DISEASE_COLORS, TREATMENT_RECOMMENDATIONS_V2, FDI_NUMBERING,
    CONFIDENCE_THRESHOLD, INFERENCE_IMGSZ
)


def estimate_tooth_number(bbox, img_w, img_h):
    """
    Panoramik röntgende bbox konumundan FDI diş numarası tahmin et.
    Panoramik röntgen düzeni (ayna - hastanın bakış açısı):
      Görüntünün solu = Hastanın sağı → Quadrant 1 (üst), Quadrant 4 (alt)
      Görüntünün sağı = Hastanın solu → Quadrant 2 (üst), Quadrant 3 (alt)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    is_upper = cy < img_h * 0.50
    norm_x = cx / img_w

    if is_upper:
        quadrant = 1 if norm_x < 0.5 else 2
    else:
        quadrant = 4 if norm_x < 0.5 else 3

    if quadrant in [1, 4]:
        tooth_pos = (0.5 - norm_x) / 0.5
    else:
        tooth_pos = (norm_x - 0.5) / 0.5

    tooth_pos = max(0, min(1, tooth_pos))
    tooth_idx = int(tooth_pos * 7) + 1
    tooth_idx = max(1, min(8, tooth_idx))

    return f"{quadrant}{tooth_idx}"


def mask_to_polygon(mask_array, simplify=True):
    """
    Binary mask'tan polygon noktaları çıkar.
    Frontend'de SVG/Canvas overlay için kullanılır.
    """
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        if simplify:
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
        pts = cnt.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


class DentalInferenceV2:
    """
    6 Sınıf Dental Inference Engine (Mask R-CNN PyTorch)
    - PyTorch Mask R-CNN modeli ile hastalık tespiti
    - Mask konturu çıkarma
    - FDI numaralama
    - Tedavi önerisi
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(DISEASE_MODEL_PATH)
        print(f"🦷 Mask R-CNN PyTorch Model yükleniyor: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 6 Sınıf + 1 Arkaplan = 7 Sınıf
        num_classes = 7
        
        # Mask R-CNN v2 mimarisini kur
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        
        # Ağırlıkları yükle (weights_only=True ile güvenli yükleme)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except Exception as e:
            print(f"⚠️ weights_only=True yüklemesinde hata oluştu, alternatif yükleme deneniyor: {e}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = DISEASE_CLASSES_V2
        print(f"✅ Model başarıyla yüklendi. Cihaz: {self.device}")

    @torch.no_grad()
    def run_inference(self, image_path, conf_threshold=None):
        """
        Tek bir görsel üzerinde inference çalıştır.
        Returns: (enhanced_img, detections_list)
        """
        if conf_threshold is None:
            conf_threshold = CONFIDENCE_THRESHOLD

        # Orijinal görseli oku
        orig_img = cv2.imread(str(image_path))
        if orig_img is None:
            raise ValueError(f"Görsel okunamadı: {image_path}")
        h, w = orig_img.shape[:2]

        # Medikal ön-işleme (CLAHE + Sharpening)
        enhanced_img = enhance_dental_image(orig_img)

        # Görseli RGB'ye çevirip tensor yap
        img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        img_tensor = F_trans.to_tensor(img_rgb).to(self.device)

        # PyTorch model tahmini
        outputs = self.model([img_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks = outputs["masks"].cpu().numpy() if "masks" in outputs else None

        detections = []
        for i in range(len(scores)):
            if scores[i] < conf_threshold:
                continue

            # Mask R-CNN tahmini arkaplanı (0) içerdiğinden sınıf id'sini 1 düşürüyoruz
            cls_id = int(labels[i]) - 1
            if cls_id < 0 or cls_id >= len(self.class_names):
                continue

            b = boxes[i]
            conf = float(scores[i])
            class_name = self.class_names[cls_id]
            fdi = estimate_tooth_number(b, w, h)

            detection = {
                "bbox": [round(float(x), 1) for x in b],
                "tooth_number": fdi,
                "tooth_name": FDI_NUMBERING.get(fdi, f"Diş #{fdi}"),
                "class": class_name,
                "class_tr": DISEASE_NAMES_TR.get(class_name, class_name),
                "confidence": round(conf, 4),
                "mask_polygons": [],
            }

            # Mask varsa kontür çıkar
            if masks is not None and i < len(masks):
                mask_array = masks[i][0] # (H, W) float mask
                binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
                # Orijinal boyuta getir
                mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                polygons = mask_to_polygon(mask_resized)
                detection["mask_polygons"] = polygons

            # Tedavi önerisi ekle
            if class_name in TREATMENT_RECOMMENDATIONS_V2:
                rec = TREATMENT_RECOMMENDATIONS_V2[class_name]
                detection["treatment"] = {
                    "tr": rec["tr"],
                    "en": rec["en"],
                    "priority": rec["priority"],
                }

            detections.append(detection)

        return enhanced_img, detections

    def visualize_results(self, img, detections, output_path=None):
        """
        Tespit sonuçlarını görsel üzerine çiz.
        """
        vis_img = img.copy()
        overlay = img.copy()

        for det in detections:
            b = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]
            fdi = det["tooth_number"]
            color = DISEASE_COLORS.get(cls, (0, 255, 0))

            # Mask konturunu çiz (yarı-şeffaf dolgu)
            for polygon in det.get("mask_polygons", []):
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color)

            # Bbox çiz
            cv2.rectangle(vis_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)

            # Etiket
            label = f"#{fdi} {DISEASE_NAMES_TR.get(cls, cls)} ({conf:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = max(int(b[1]) - 8, label_size[1] + 4)
            
            # Etiket arkaplanı
            cv2.rectangle(
                vis_img,
                (int(b[0]), label_y - label_size[1] - 4),
                (int(b[0]) + label_size[0] + 4, label_y + 4),
                color, -1
            )
            cv2.putText(
                vis_img, label,
                (int(b[0]) + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Mask overlay'i yarı-şeffaf olarak birleştir
        vis_img = cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0)

        if output_path:
            cv2.imwrite(str(output_path), vis_img)
            print(f"   💾 Kaydedildi: {output_path}")

        return vis_img

    def generate_clinical_report(self, detections, image_name=""):
        """
        Tespitlerden yapısal klinik rapor oluştur.
        """
        tooth_map = {}
        for det in detections:
            fdi = det["tooth_number"]
            if fdi not in tooth_map:
                tooth_map[fdi] = {
                    "tooth_number": fdi,
                    "tooth_name": det["tooth_name"],
                    "findings": []
                }
            tooth_map[fdi]["findings"].append({
                "disease": det["class"],
                "disease_tr": det["class_tr"],
                "confidence": det["confidence"],
                "treatment": det.get("treatment", {}),
                "bbox": det["bbox"],
                "mask_polygons": det["mask_polygons"],
            })

        total_findings = len(detections)
        affected_teeth = len(tooth_map)
        high_priority = sum(
            1 for d in detections
            if d.get("treatment", {}).get("priority") == "yüksek"
        )

        report = {
            "image": image_name,
            "summary": {
                "total_findings": total_findings,
                "affected_teeth": affected_teeth,
                "high_priority_count": high_priority,
                "summary_text": (
                    f"Analiz sonucunda {affected_teeth} adet dişte toplam {total_findings} bulgu tespit edilmiştir. "
                    f"{high_priority} adet yüksek öncelikli bulgu mevcuttur."
                    if total_findings > 0 else
                    "Görsel analizde belirgin patolojik bulgu saptanmamıştır."
                ),
            },
            "teeth": list(tooth_map.values()),
            "all_detections": detections,
            "disclaimer": (
                "⚠️ Bu rapor yapay zeka destekli ön değerlendirme niteliğindedir. "
                "Kesin teşhis ve tedavi planı için klinik muayene ve hekim değerlendirmesi şarttır."
            ),
        }
        return report
