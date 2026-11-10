"""
Huấn luyện mô hình ShuttlecockTrackerNet.
    python src/shuttlecock_tracker/train.py
hoặc:
    python -m src.shuttlecock_tracker.train
"""
import os
import sys
from pathlib import Path
import random
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Tự động nhận diện thư mục gốc của dự án để chạy lệnh ở bất kỳ đâu cũng không bị lỗi import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shuttlecock_tracker.dataset import BadmintonDataset
from src.shuttlecock_tracker.loss import get_loss_fn
from src.shuttlecock_tracker.model import ShuttlecockTrackerNet


# ==============================================================================
# CẤU HÌNH THAM SỐ
# ==============================================================================
# Bật chế độ TEST_MODE để chạy thử
TEST_MODE = True            # Đổi False khi training
MAX_TRAIN_SAMPLES = 20
MAX_VAL_SAMPLES = 10

NUM_EPOCHS = 1 if TEST_MODE else 30
BATCH_SIZE = 2
LEARNING_RATE = 1e-3        # Tốc độ học khởi điểm của Optimizer
POS_WEIGHT = 100.0          # Trọng số phạt vùng quả cầu trong BCE Loss

DATA_DIR = "Dataset"
WEIGHTS_DIR = "weights"
BEST_WEIGHTS_FILE = "shuttlecock_unet.pth"
LAST_WEIGHTS_FILE = "shuttlecock_lastmodel.pth"

# Nếu muốn tiếp tục huấn luyện từ checkpoint cũ, điền đường dẫn vào đây (ví dụ: "weights/shuttlecock_lastmodel.pth")
RESUME_CHECKPOINT = None
# ==============================================================================


def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên để kết quả ổn định qua các lần chạy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_coords_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, float, float]:
    """Tìm đỉnh cực đại (Peak) trên heatmap dự đoán để lấy tọa độ (x, y)."""
    max_val = float(np.max(heatmap))
    if max_val < threshold:
        return 0, 0.0, 0.0
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return 1, float(x), float(y)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """Vòng lặp huấn luyện trong 1 Epoch."""
    model.train()  # Bật chế độ huấn luyện
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for frames, heatmaps, _ in pbar:
        frames = frames.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()                         # 1. Xóa gradient cũ
        logits = model(frames, return_logits=True)     # 2. Dự đoán logits thô
        loss = loss_fn(logits, heatmaps)               # 3. Tính hàm mất mát có trọng số
        loss.backward()                                # 4. Lan truyền ngược tính gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 5. Cắt tỉa gradient
        optimizer.step()                               # 6. Cập nhật trọng số

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    input_w: int = 512,
    input_h: int = 288,
    dist_tolerance: float = 5.0
) -> Dict[str, float]:
    """Vòng lặp đánh giá trên tập Test."""
    model.eval()  # Chuyển sang chế độ đánh giá
    total_loss = 0.0
    tp, fp, fn = 0, 0, 0

    pbar = tqdm(dataloader, desc="[Validating on Test set]", leave=False)
    for frames, heatmaps, metadata in pbar:
        frames = frames.to(device)
        heatmaps = heatmaps.to(device)

        logits = model(frames, return_logits=True)
        loss = loss_fn(logits, heatmaps)
        total_loss += loss.item()

        # Chuyển logits sang xác suất [0.0, 1.0] bằng Sigmoid
        probs = torch.sigmoid(logits).cpu().numpy()

        batch_size = frames.size(0)
        for b in range(batch_size):
            # Lấy heatmap của frame giữa (frame t, index channel = 1)
            pred_hm = probs[b, 1]
            vis_pred, px, py = extract_coords_from_heatmap(pred_hm, threshold=0.5)

            # Lấy nhãn ground truth của frame giữa
            gt_center = metadata["labels"][1]
            vis_gt = int(gt_center["vis"][b])
            gt_x = float(gt_center["x"][b])
            gt_y = float(gt_center["y"][b])

            orig_w = float(metadata["orig_size"][0][b]) or 1280.0
            orig_h = float(metadata["orig_size"][1][b]) or 720.0

            # Quy đổi tọa độ dự đoán về kích thước video gốc
            scaled_pred_x = px * (orig_w / float(input_w))
            scaled_pred_y = py * (orig_h / float(input_h))

            if vis_gt == 1:
                if vis_pred == 1:
                    dist = np.hypot(scaled_pred_x - gt_x, scaled_pred_y - gt_y)
                    if dist <= dist_tolerance:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
            else:
                if vis_pred == 1:
                    fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "val_loss": total_loss / max(1, len(dataloader)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--> Thiết bị tính toán được sử dụng: {device}")
    if TEST_MODE:
        print("--> [CHẾ ĐỘ TEST NHANH]: Bật giới hạn mẫu để chạy mượt trên CPU yếu.")

    # 1. Tạo thư mục lưu weights
    weights_path = Path(WEIGHTS_DIR)
    weights_path.mkdir(parents=True, exist_ok=True)
    best_weights_path = weights_path / BEST_WEIGHTS_FILE
    last_weights_path = weights_path / LAST_WEIGHTS_FILE

    # 2. Khởi tạo Dataset Train và Test
    print(f"--> Đang quét tập Train (Amateur + Professional) trong '{DATA_DIR}'...")
    train_dataset = BadmintonDataset(
        dataset_root=Path(DATA_DIR),
        subsets=["Amateur", "Professional"],
        input_width=512,
        input_height=288,
        sigma=2.5,
        max_samples=MAX_TRAIN_SAMPLES if TEST_MODE else None
    )
    print(f"    [+] Số mẫu nạp vào tập Train: {len(train_dataset)}")

    print(f"--> Đang quét tập Validation (Test: match1, match2, match3)...")
    val_dataset = BadmintonDataset(
        dataset_root=Path(DATA_DIR),
        subsets=["Test"],
        input_width=512,
        input_height=288,
        sigma=2.5,
        max_samples=MAX_VAL_SAMPLES if TEST_MODE else None
    )
    print(f"    [+] Số mẫu nạp vào tập Validation: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Lỗi: Không tìm thấy dữ liệu! Hãy kiểm tra lại thư mục Dataset.")
        return

    # Tự động dùng 2 luồng đọc dữ liệu nền trên GPU/Linux để tăng tốc tối đa
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 3. Khởi tạo Mô hình, Hàm Loss (criterion), Bộ tối ưu và Lịch giảm LR
    model = ShuttlecockTrackerNet(in_channels=9, out_channels=3).to(device)
    loss_fn = get_loss_fn(loss_name="weighted_bce", pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch = 1
    best_f1 = -1.0

    # 4. Nạp checkpoint cũ nếu có
    if RESUME_CHECKPOINT and Path(RESUME_CHECKPOINT).exists():
        print(f"--> Đang nạp lại trọng số cũ từ: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        print(f"--> Tiếp tục từ Epoch {start_epoch} (Best F1 trước đó: {best_f1:.4f})")

    # 5. Vòng lặp huấn luyện chính
    print("\n" + "=" * 65)
    print(" BẮT ĐẦU HUẤN LUYỆN SHUTTLECOCK TRACKER")
    print(f"  + Số lượng Epochs:       {NUM_EPOCHS}")
    print(f"  + Kích thước Batch Size: {BATCH_SIZE}")
    print(f"  + File lưu tốt nhất:     {best_weights_path}")
    print(f"  + File lưu gần nhất:     {last_weights_path}")
    print("=" * 65)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Huấn luyện 1 epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)

        # Đánh giá trên tập Test
        val_metrics = validate(model, val_loader, loss_fn, device, input_w=512, input_h=288)
        val_loss = val_metrics["val_loss"]
        precision = val_metrics["precision"]
        recall = val_metrics["recall"]
        f1 = val_metrics["f1"]

        # Cập nhật lịch học
        scheduler.step(val_loss)

        # In kết quả Epoch
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}"
        )

        # Lưu trọng số gần nhất
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1": f1,
            "best_f1": max(best_f1, f1)
        }, last_weights_path)

        # Lưu trọng số tốt nhất khi F1 cải thiện
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "val_loss": val_loss
            }, best_weights_path)
            print(f"  [+] Đã cập nhật trọng số tốt nhất -> {best_weights_path} (F1: {best_f1:.4f})")

    print("\n" + "=" * 65)
    print(" CHẠY THÀNH CÔNG!")
    print(f"  - Trọng số tốt nhất đã lưu tại: {best_weights_path}")
    print(f"  - Trọng số gần nhất đã lưu tại: {last_weights_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
