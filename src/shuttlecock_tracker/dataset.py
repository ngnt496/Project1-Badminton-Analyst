"""
Dataset loader và hàm tạo Gaussian Heatmap cho bài toán nhận diện quả cầu lông.
Sử dụng cơ chế Lazy Loading (chỉ nạp 3 frame cần thiết vào RAM khi được gọi).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def generate_gaussian_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float = 2.5
) -> np.ndarray:
    """
    Tạo bản đồ nhiệt Gaussian 2D tại vị trí (center_x, center_y).
    - Tâm tại vị trí quả cầu có giá trị cực đại = 1.0.
    - Lan tỏa mờ dần về 0.0 theo bán kính 3 * sigma.
    - Trả về ma trận numpy kích thước (height, width).
    """
    # Nếu tọa độ nằm ngoài biên ảnh, trả về ma trận toàn số 0.0
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
        return np.zeros((height, width), dtype=np.float32)

    # Giới hạn vùng tính toán trong hộp bán kính 3*sigma để tối ưu CPU
    radius = int(np.ceil(3 * sigma))
    x0 = max(0, int(center_x - radius))
    x1 = min(width, int(center_x + radius + 1))
    y0 = max(0, int(center_y - radius))
    y1 = min(height, int(center_y + radius + 1))

    heatmap = np.zeros((height, width), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return heatmap

    # Tính toán giá trị Gaussian theo khoảng cách Euclidean
    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma))

    heatmap[y0:y1, x0:x1] = gaussian
    return heatmap


class BadmintonDataset(Dataset):
    """
    - Khởi tạo: Chỉ lập danh mục chỉ mục (Index) từ các file CSV (~ vài MB RAM).
    - __getitem__: Chỉ đọc đúng 3 frame từ mp4 vào RAM khi mô hình yêu cầu một mẫu.
    """
    def __init__(
        self,
        dataset_root: Path = Path("Dataset"),
        subsets: List[str] = ("Amateur", "Professional"),
        input_width: int = 512,
        input_height: int = 288,
        sigma: float = 2.5,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.subsets = subsets
        self.input_width = input_width
        self.input_height = input_height
        self.sigma = sigma

        # Danh sách metadata: Chỉ lưu đường dẫn video, số thứ tự frame và tọa độ x, y
        self.samples: List[Dict] = []
        self._index_dataset()

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def _index_dataset(self):
        """
        Quét nhanh qua các file CSV và kiểm tra video mp4 tương ứng.
        Chỉ lưu thông tin tọa độ và đường dẫn vào RAM.
        """
        for subset in self.subsets:
            subset_dir = self.dataset_root / subset
            if not subset_dir.exists():
                continue

            for match_dir in sorted(subset_dir.glob("match*")):
                video_dir = match_dir / "video"
                csv_dir = match_dir / "csv"
                if not video_dir.exists() or not csv_dir.exists():
                    continue

                for csv_path in sorted(csv_dir.glob("*_ball.csv")):
                    clip_name = csv_path.stem.replace("_ball", "")
                    video_path = video_dir / f"{clip_name}.mp4"
                    if not video_path.exists():
                        continue

                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        continue

                    df.columns = [c.strip() for c in df.columns]
                    if not all(col in df.columns for col in ["Frame", "Visibility", "X", "Y"]):
                        continue

                    num_frames = len(df)
                    # Tạo bộ ba frame liên tiếp (i-1, i, i+1) với bước nhảy stride=2 để dữ liệu cô đọng và chạy nhanh gấp đôi
                    stride = 2
                    for i in range(1, num_frames - 1, stride):
                        self.samples.append({
                            "video_path": str(video_path),
                            "frame_indices": (i - 1, i, i + 1),
                            "labels": [
                                {
                                    "vis": int(df.iloc[k]["Visibility"]),
                                    "x": float(df.iloc[k]["X"]),
                                    "y": float(df.iloc[k]["Y"])
                                }
                                for k in (i - 1, i, i + 1)
                            ]
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def _read_single_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        """Đọc đúng 1 frame chỉ định từ video mp4."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        # Chuyển hệ màu OpenCV BGR sang RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Hàm này chỉ được gọi khi mô hình cần dữ liệu:
        1. Mở video, đọc đúng 3 frame (i-1, i, i+1).
        2. Resize về 288x512 và chuẩn hóa về [0, 1].
        3. Tạo 3 Heatmap nhãn từ tọa độ (X, Y).
        4. Trả về Tensor cho batch hiện tại.
        """
        sample = self.samples[idx]
        video_path = sample["video_path"]
        f_indices = sample["frame_indices"]
        labels = sample["labels"]

        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        scale_x = self.input_width / float(orig_w)
        scale_y = self.input_height / float(orig_h)

        resized_frames = []
        heatmaps = []

        for f_idx, label in zip(f_indices, labels):
            frame = self._read_single_frame(cap, f_idx)
            if frame is None:
                # Nếu video bị lỗi ở frame này, thay bằng khung đen
                frame = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

            # Chuẩn hóa giá trị pixel về [0.0, 1.0]
            resized_frames.append(frame.astype(np.float32) / 255.0)

            # Sinh Gaussian Heatmap nhãn
            if label["vis"] == 1 and label["x"] > 0 and label["y"] > 0:
                cx = label["x"] * scale_x
                cy = label["y"] * scale_y
                hm = generate_gaussian_heatmap(
                    height=self.input_height,
                    width=self.input_width,
                    center_x=cx,
                    center_y=cy,
                    sigma=self.sigma
                )
            else:
                # Quả cầu bị khuất hoặc ngoài sân -> ma trận toàn 0
                hm = np.zeros((self.input_height, self.input_width), dtype=np.float32)

            heatmaps.append(hm)

        # Đóng video ngay sau khi lấy xong 3 frame để giải phóng file descriptor
        cap.release()

        # Ghép 3 frames theo trục kênh: (288, 512, 3) x 3 -> (288, 512, 9) -> (9, 288, 512)
        frames_concat = np.concatenate(resized_frames, axis=2)
        frames_tensor = torch.from_numpy(frames_concat).permute(2, 0, 1).float()

        # Xếp 3 heatmaps: (3, 288, 512)
        heatmaps_tensor = torch.from_numpy(np.stack(heatmaps, axis=0)).float()

        metadata = {
            "video_path": video_path,
            "center_frame": f_indices[1],
            "orig_size": (orig_w, orig_h),
            "labels": labels
        }

        return frames_tensor, heatmaps_tensor, metadata


if __name__ == "__main__":
    # Test thử trực tiếp trên dữ liệu thật
    print("Testing BadmintonDataset...")
    dataset = BadmintonDataset(
        dataset_root=Path("Dataset"),
        subsets=["Amateur"],
        input_width=512,
        input_height=288,
        sigma=2.5,
        max_samples=10  # Chỉ test 10 mẫu đầu tiên
    )
    print(f"Tổng số mẫu tìm thấy trong chỉ mục: {len(dataset)}")
    
    if len(dataset) > 0:
        frames, heatmaps, meta = dataset[0]
        print(f"Frames Tensor Shape:   {frames.shape}  (Kỳ vọng: [9, 288, 512])")
        print(f"Heatmaps Tensor Shape: {heatmaps.shape} (Kỳ vọng: [3, 288, 512])")
        print(f"Dải giá trị Frames:   min={frames.min():.2f}, max={frames.max():.2f}")
        print(f"Dải giá trị Heatmaps: min={heatmaps.min():.2f}, max={heatmaps.max():.2f}")
        print("Xác thực Dataset thành công!")
