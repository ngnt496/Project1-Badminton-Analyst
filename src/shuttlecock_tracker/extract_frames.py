"""
Script trích xuất toàn bộ video MP4 thành các file ảnh JPG đã resize (512x288).
Giải quyết triệt để vấn đề nghẽn cổ chai I/O khi đọc video trong lúc huấn luyện.
Chạy 1 lần duy nhất trên Colab (mất khoảng 3-4 phút).
"""
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_all_videos(
    dataset_root: str = "Dataset",
    output_root: str = "Dataset_frames",
    target_width: int = 512,
    target_height: int = 288,
    quality: int = 95
):
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(dataset_path.glob("*/*/video/*.mp4")))
    print(f"--> Tìm thấy tổng cộng {len(video_files)} video clips cần trích xuất.")

    for v_path in tqdm(video_files, desc="Trích xuất frames"):
        # Cấu trúc: Dataset_frames/<subset>/<match>/<clip_name>/frame_%04d.jpg
        rel_parts = v_path.relative_to(dataset_path).parts
        subset = rel_parts[0]       # Amateur / Professional / Test
        match_name = rel_parts[1]   # match1, match2...
        clip_name = v_path.stem     # 1_00_01

        save_dir = output_path / subset / match_name / clip_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        frame_idx = 0

        # Đọc tuần tự từ đầu đến cuối (Sequential Read cực nhanh, ~200-300 fps)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            out_img_path = save_dir / f"{frame_idx:04d}.jpg"
            if not out_img_path.exists():
                resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(out_img_path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])

            frame_idx += 1

        cap.release()

    print(f"\n[+] TRÍCH XUẤT HOÀN TẤT! Dữ liệu ảnh đã sẵn sàng tại '{output_root}'")


if __name__ == "__main__":
    extract_all_videos()
