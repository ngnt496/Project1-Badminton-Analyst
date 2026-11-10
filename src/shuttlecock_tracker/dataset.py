"""
Dataset loader và hàm tạo Gaussian Heatmap cho bài toán nhận diện quả cầu lông.
Hỗ trợ cả 2 chế độ:
1. Đọc trực tiếp từ ảnh JPG đã trích xuất sẵn trong 'Dataset_frames/' (CỰC NHANH, khuyến nghị cho Colab/GPU).
2. Tự động fallback đọc từ file '.mp4' nếu chưa trích xuất ảnh.
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
    """Tạo bản đồ nhiệt Gaussian 2D tại vị trí (center_x, center_y)."""
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
        return np.zeros((height, width), dtype=np.float32)

    radius = int(np.ceil(3 * sigma))
    x0 = max(0, int(center_x - radius))
    x1 = min(width, int(center_x + radius + 1))
    y0 = max(0, int(center_y - radius))
    y1 = min(height, int(center_y + radius + 1))

    heatmap = np.zeros((height, width), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return heatmap

    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma))

    heatmap[y0:y1, x0:x1] = gaussian
    return heatmap


class BadmintonDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path = Path("Dataset"),
        frames_root: Path = Path("Dataset_frames"),
        subsets: List[str] = ("Amateur", "Professional"),
        input_width: int = 512,
        input_height: int = 288,
        sigma: float = 2.5,
        max_samples: Optional[int] = None,
        stride: int = 1
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.frames_root = Path(frames_root)
        self.subsets = subsets
        self.input_width = input_width
        self.input_height = input_height
        self.sigma = sigma
        self.stride = stride

        self.samples: List[Dict] = []
        self._index_dataset()

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def _index_dataset(self):
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
                    clip_frames_dir = self.frames_root / subset / match_dir.name / clip_name

                    # Hợp lệ nếu có thư mục ảnh JPG hoặc có file video MP4
                    if not clip_frames_dir.exists() and not video_path.exists():
                        continue

                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        continue

                    df.columns = [c.strip() for c in df.columns]
                    if not all(col in df.columns for col in ["Frame", "Visibility", "X", "Y"]):
                        continue

                    num_frames = len(df)
                    # Tạo bộ ba frame liên tiếp (i-1, i, i+1) với bước nhảy stride
                    for i in range(1, num_frames - 1, self.stride):
                        self.samples.append({
                            "subset": subset,
                            "match": match_dir.name,
                            "clip_name": clip_name,
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

    def _read_frame(
        self,
        sample: Dict,
        frame_idx: int,
        cap: Optional[cv2.VideoCapture] = None
    ) -> Optional[np.ndarray]:
        """
        Ưu tiên đọc từ file ảnh JPG (siêu nhanh: ~0.001s).
        Nếu chưa trích xuất ảnh, tự động fallback đọc từ VideoCapture.
        """
        subset = sample["subset"]
        match_name = sample["match"]
        clip_name = sample["clip_name"]

        # 1. Kiểm tra ảnh đã trích xuất trong Dataset_frames/
        img_path = self.frames_root / subset / match_name / clip_name / f"{frame_idx:04d}.jpg"
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Fallback: Đọc từ video MP4 nếu chưa có thư mục ảnh
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return cv2.resize(frame_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        f_indices = sample["frame_indices"]
        labels = sample["labels"]

        # Kiểm tra xem có thư mục ảnh JPG không
        subset = sample["subset"]
        match_name = sample["match"]
        clip_name = sample["clip_name"]
        clip_frames_dir = self.frames_root / subset / match_name / clip_name
        use_images = clip_frames_dir.exists()

        cap = None
        orig_w, orig_h = 1280, 720
        if not use_images:
            cap = cv2.VideoCapture(sample["video_path"])
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        scale_x = self.input_width / float(orig_w)
        scale_y = self.input_height / float(orig_h)

        resized_frames = []
        heatmaps = []

        for f_idx, label in zip(f_indices, labels):
            frame = self._read_frame(sample, f_idx, cap=cap)
            if frame is None:
                frame = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)

            resized_frames.append(frame.astype(np.float32) / 255.0)

            # Tạo Heatmap nhãn
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
                hm = np.zeros((self.input_height, self.input_width), dtype=np.float32)

            heatmaps.append(hm)

        if cap is not None:
            cap.release()

        frames_concat = np.concatenate(resized_frames, axis=2)
        frames_tensor = torch.from_numpy(frames_concat).permute(2, 0, 1).float()
        heatmaps_tensor = torch.from_numpy(np.stack(heatmaps, axis=0)).float()

        metadata = {
            "video_path": sample["video_path"],
            "center_frame": f_indices[1],
            "orig_size": (orig_w, orig_h),
            "labels": labels
        }

        return frames_tensor, heatmaps_tensor, metadata


if __name__ == "__main__":
    print("Testing BadmintonDataset...")
    ds = BadmintonDataset(max_samples=10)
    print(f"Total samples: {len(ds)}")
    if len(ds) > 0:
        f, h, m = ds[0]
        print(f"Frames: {f.shape}, Heatmaps: {h.shape}")
        print("Dataset ready!")
