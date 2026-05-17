import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F

# ═══════════════════════════════════════════════════════════════
# 1. Dataset Sınıfı (YOLO Formatını Mask R-CNN Formatına Çeviren)
# ═══════════════════════════════════════════════════════════════
class DentalMaskDatasetV3(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.transforms = transforms
        self.imgs = sorted([p for p in self.img_dir.glob("*.jpg")])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # Görseli oku ve RGB'ye çevir
        img_cv = cv2.imread(str(img_path))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_cv.shape[:2]

        boxes = []
        labels = []
        masks = []

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                cls_id = int(parts[0])
                # Mask R-CNN expects labels >= 1 (0 is background)
                cls_id += 1  

                pts = [float(x) for x in parts[1:]]
                poly_pts = []
                for i in range(0, len(pts), 2):
                    x = max(0.0, min(1.0, pts[i])) * w
                    y = max(0.0, min(1.0, pts[i+1])) * h
                    poly_pts.append([x, y])

                if len(poly_pts) < 3:
                    continue

                # Poligondan Binary Mask oluştur
                poly_np = np.array([poly_pts], dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, poly_np, 1)

                # Bounding Box bul (x_min, y_min, x_max, y_max)
                pos = np.where(mask)
                if len(pos[0]) == 0:
                    continue
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                # Geçersiz box'ları ele
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
                masks.append(mask)

        # Tensörlere çevir
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Boş görsel için dummy tensor (hata vermemesi için)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

        target["image_id"] = torch.tensor([idx])
        
        # Resmi tensor'a çevir (C, H, W formatına ve 0-1 aralığına)
        img_tensor = F.to_tensor(img_cv)

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ═══════════════════════════════════════════════════════════════
# 2. Model Tanımı
# ═══════════════════════════════════════════════════════════════
def get_model_instance_segmentation(num_classes):
    # COCO verisiyle eğitilmiş hazır modeli yükle
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)

    # Box tahmini katmanını değiştir
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask tahmini katmanını değiştir
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# ═══════════════════════════════════════════════════════════════
# 3. Eğitim Döngüsü
# ═══════════════════════════════════════════════════════════════
def main():
    print("🦷 Dental AI v3 - Mask R-CNN Eğitim Başlıyor")
    print("=" * 60)

    # Dizinler
    DATASET_DIR = Path("../dataset")
    TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
    TRAIN_LBL_DIR = DATASET_DIR / "labels" / "train"
    VAL_IMG_DIR = DATASET_DIR / "images" / "val"
    VAL_LBL_DIR = DATASET_DIR / "labels" / "val"
    
    OUT_DIR = Path("./weights")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_IMG_DIR.exists():
        print("❌ HATA: Dataset bulunamadı! Önce 'prepare_maskrcnn_v3.py' çalıştırın.")
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🖥️ Donanım: {device}")

    # Sınıf sayısı: 6 Hastalık + 1 Arkaplan (Background) = 7
    num_classes = 7
    
    # Dataset ve Dataloader
    dataset_train = DentalMaskDatasetV3(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    dataset_val = DentalMaskDatasetV3(VAL_IMG_DIR, VAL_LBL_DIR)

    # VRAM taşmaması için Batch Size = 2 (RTX 3050 4GB için ideal)
    data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"📂 Eğitim Görseli: {len(dataset_train)}")
    print(f"📂 Validasyon Görseli: {len(dataset_val)}")

    # Modeli Yükle
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer ve Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    best_loss = float('inf')

    print("\n🚀 Eğitim Başlıyor...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for i, (images, targets) in enumerate(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if i % 50 == 0:
                print(f"  Epoch: {epoch+1} | Batch: {i}/{len(data_loader_train)} | Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(data_loader_train)
        epoch_time = time.time() - start_time
        
        print(f"\n✅ Epoch {epoch+1} Tamamlandı. Ortalama Kayıp (Loss): {avg_loss:.4f} | Süre: {epoch_time:.0f}sn")

        # Basit Validasyon (Sadece Model Loss Kontrolü)
        with torch.no_grad():
            val_loss = 0
            for images, targets in data_loader_val:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
            
            avg_val_loss = val_loss / len(data_loader_val)
            print(f"📊 Validasyon Kaybı (Loss): {avg_val_loss:.4f}")

        # En iyi modeli kaydet
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = OUT_DIR / "best_maskrcnn.pth"
            torch.save(model.state_dict(), str(save_path))
            print(f"💾 Yeni en iyi model kaydedildi! -> {save_path}\n")
        else:
            print("\n")

    print("🎉 Eğitim Tamamen Bitti!")
    print(f"En iyi model ağırlığı: {OUT_DIR}/best_maskrcnn.pth")

if __name__ == "__main__":
    main()
