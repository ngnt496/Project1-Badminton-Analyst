"""
Dự đoán và trực quan hóa đường bay quả cầu lông trên video thực tế.
    python src/shuttlecock_tracker/predict.py
hoặc:
    python src/shuttlecock_tracker/predict.py --video Dataset/Test/match1/video/1_05_02.mp4 --weights weights/shuttlecock_unet.pth
"""
import argparse
from collections import deque
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shuttlecock_tracker.model import ShuttlecockTrackerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton Shuttlecock Tracker - Video Prediction")
    parser.add_argument(
        "--video",
        type=str,
        default="Dataset/Test/match1/video/1_05_02.mp4",
        help="Đường dẫn đến file video mp4 cần dự đoán"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/shuttlecock_unet.pth",
        help="Đường dẫn file trọng số đã huấn luyện (.pth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Thư mục lưu video kết quả và file CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Ngưỡng xác suất heatmap (0.0 đến 1.0) để kết luận có quả cầu"
    )
    parser.add_argument(
        "--tail_length",
        type=int,
        default=10,
        help="Độ dài vệt đuôi quỹ đạo quả cầu (số frame)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size khi suy luận AI (4 hoặc 8 tối ưu cho CPU)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Giới hạn số frame đầu tiên để test nhanh (để trống nếu muốn chạy hết video)"
    )
    return parser.parse_args()


def load_model(weights_path: Path, device: torch.device) -> ShuttlecockTrackerNet:
    """Khởi tạo mô hình và nạp trọng số checkpoint."""
    model = ShuttlecockTrackerNet(in_channels=9, out_channels=3)
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weights_path}")

    print(f"--> Đang nạp trọng số từ: {weights_path}")
    checkpoint = torch.load(str(weights_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        f1 = checkpoint.get("f1", "N/A")
        print(f"    [+] Trọng số nạp thành công! (Epoch: {epoch} | Val F1: {f1})")
    else:
        model.load_state_dict(checkpoint)
        print("    [+] Trọng số nạp thành công!")

    model.to(device)
    model.eval()
    return model


def extract_coordinates(heatmap: np.ndarray, threshold: float = 0.45) -> Tuple[int, float, float, float]:
    """Tìm điểm cực đại (Peak) trên heatmap xác suất."""
    max_val = float(np.max(heatmap))
    if max_val < threshold:
        return 0, 0.0, 0.0, max_val

    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return 1, float(x), float(y), max_val


def draw_hud_panel(frame: np.ndarray, text_lines: List[Tuple[str, Tuple[int, int, int]]]):
    """Vẽ bảng dashboard thông số bán trong suốt ở góc màn hình."""
    panel_w = 340
    panel_h = 24 + len(text_lines) * 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (0, 230, 115), 2)

    for i, (text, color) in enumerate(text_lines):
        y_pos = 40 + i * 26
        cv2.putText(frame, text, (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def run_prediction(args):
    # Tối ưu hóa số luồng CPU cho PyTorch nếu chạy trên CPU
    if not torch.cuda.is_available():
        threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(threads)

    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Thiết bị sử dụng: {device.type.upper()}")

    # 1. Nạp Model
    model = load_model(weights_path, device)

    # 2. Mở Video & Đọc thông số
    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = args.max_frames if args.max_frames else total_frames
    print(f"--> Đang xử lý video: {video_path.name}")
    print(f"    - Độ phân giải: {orig_w}x{orig_h} | FPS: {fps:.1f} | Tổng số frame cần xử lý: {max_frames}")

    # GIAI ĐOẠN 1: Đọc tuần tự các frame ở kích thước chuẩn (512, 288) dạng uint8 (tiết kiệm RAM)
    print("\n[Bước 1/3] Đọc và tiền xử lý khung hình...")
    input_w, input_h = 512, 288
    resized_frames_uint8 = []

    frame_count = 0
    pbar_read = tqdm(total=max_frames, desc="  - Đang đọc video", unit="frame")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        resized_frames_uint8.append(resized)
        frame_count += 1
        pbar_read.update(1)
    pbar_read.close()

    n_frames = len(resized_frames_uint8)
    if n_frames == 0:
        cap.release()
        print("Lỗi: Không đọc được khung hình nào từ video!")
        return

    # GIAI ĐOẠN 2: Suy luận qua AI Model (Triplets)
    print(f"\n[Bước 2/3] Suy luận AI nhận diện quả cầu lông ({n_frames} frames)...")
    scale_x = float(orig_w) / float(input_w)
    scale_y = float(orig_h) / float(input_h)

    predictions = []
    batch_size = max(1, args.batch_size)
    start_infer_time = time.time()

    pbar_infer = tqdm(total=n_frames, desc="  - AI Inferencing", unit="frame")
    with torch.no_grad():
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch_triplets = []

            for i in range(start_idx, end_idx):
                idx_prev = max(0, i - 1)
                idx_curr = i
                idx_next = min(n_frames - 1, i + 1)

                f0 = resized_frames_uint8[idx_prev].astype(np.float32) / 255.0
                f1 = resized_frames_uint8[idx_curr].astype(np.float32) / 255.0
                f2 = resized_frames_uint8[idx_next].astype(np.float32) / 255.0

                triplet = np.concatenate([f0, f1, f2], axis=2).transpose(2, 0, 1)
                batch_triplets.append(triplet)

            batch_tensor = torch.from_numpy(np.stack(batch_triplets, axis=0)).to(device)
            outputs = model(batch_tensor, return_logits=False)
            outputs_np = outputs.cpu().numpy()

            for b in range(outputs_np.shape[0]):
                hm_curr = outputs_np[b, 1]  # Frame hiện tại ở giữa
                vis, px, py, conf = extract_coordinates(hm_curr, threshold=args.threshold)

                if vis == 1:
                    orig_x = px * scale_x
                    orig_y = py * scale_y
                else:
                    orig_x, orig_y = 0.0, 0.0

                predictions.append((vis, orig_x, orig_y, conf))

            pbar_infer.update(end_idx - start_idx)

    pbar_infer.close()
    del resized_frames_uint8

    infer_elapsed = time.time() - start_infer_time
    infer_fps = n_frames / infer_elapsed if infer_elapsed > 0 else 0
    print(f"    [+] Thời gian suy luận: {infer_elapsed:.2f}s (~{infer_fps:.1f} FPS)")

    # Lưu file CSV kết quả
    csv_filename = output_dir / f"{video_path.stem}_ball_predict.csv"
    csv_data = []
    for f_idx, (vis, x, y, conf) in enumerate(predictions):
        csv_data.append({
            "Frame": f_idx,
            "Visibility": vis,
            "X": int(round(x)) if vis == 1 else 0,
            "Y": int(round(y)) if vis == 1 else 0
        })
    df_pred = pd.DataFrame(csv_data)
    df_pred.to_csv(csv_filename, index=False)
    print(f"    [+] Đã lưu file tọa độ CSV: {csv_filename}")

    # GIAI ĐOẠN 3: Xuất video có vẽ quỹ đạo cầu
    print(f"\n[Bước 3/3] Xuất video vẽ quỹ đạo đường cầu...")
    out_video_filename = output_dir / f"{video_path.stem}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_filename), fourcc, fps, (orig_w, orig_h))

    # Tua lại video từ frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trajectory_queue = deque(maxlen=args.tail_length)

    detected_count = sum(p[0] for p in predictions)
    detect_rate = (detected_count / n_frames) * 100 if n_frames > 0 else 0

    pbar_render = tqdm(total=n_frames, desc="  - Render Video", unit="frame")
    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        vis, x, y, conf = predictions[frame_idx]

        if vis == 1:
            trajectory_queue.append((int(x), int(y)))
        else:
            trajectory_queue.append(None)

        # Vẽ vệt đuôi quỹ đạo
        pts = list(trajectory_queue)
        for j in range(1, len(pts)):
            if pts[j - 1] is None or pts[j] is None:
                continue
            alpha = (j + 1) / len(pts)
            thickness = max(2, int(6 * alpha))
            # Màu chuyển sắc từ xanh lá neon sang vàng rực rỡ
            color = (
                int(20 * (1 - alpha) + 0 * alpha),
                int(255 * (1 - alpha) + 230 * alpha),
                int(80 * (1 - alpha) + 255 * alpha)
            )
            cv2.line(frame, pts[j - 1], pts[j], color, thickness)

        # Vẽ vị trí quả cầu tại frame hiện tại
        if vis == 1:
            cx, cy = int(x), int(y)
            cv2.circle(frame, (cx, cy), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            coord_str = f"({cx}, {cy})"
            cv2.putText(frame, coord_str, (cx + 12, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Bảng HUD hiển thị
        hud_info = [
            (f"Frame: {frame_idx + 1}/{n_frames}", (255, 255, 255)),
            (f"Shuttle: {'DETECTED' if vis == 1 else 'NOT FOUND'}",
             (0, 255, 100) if vis == 1 else (0, 80, 255)),
            (f"Confidence: {conf:.2f}", (0, 220, 255)),
            (f"Detection Rate: {detect_rate:.1f}%", (200, 200, 200))
        ]
        draw_hud_panel(frame, hud_info)

        writer.write(frame)
        pbar_render.update(1)

    pbar_render.close()
    cap.release()
    writer.release()

    print("\n" + "=" * 65)
    print(" DỰ ĐOÁN VÀ TRỰC QUAN HÓA THÀNH CÔNG!")
    print(f"  - File Video: {out_video_filename}")
    print(f"  - File CSV:   {csv_filename}")
    print(f"  - Tổng frame: {n_frames} | Số frame phát hiện cầu: {detected_count} ({detect_rate:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    args = parse_args()
    run_prediction(args)
