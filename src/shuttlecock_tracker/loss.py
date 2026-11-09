"""
Module định nghĩa các hàm mất mát (Loss Functions) cho bài toán phát hiện quả cầu lông qua Heatmap.
Bao gồm:
1. WeightedBCEWithLogitsLoss: Binary Cross Entropy có trọng số bù trừ mất cân bằng pixel.
2. ModifiedFocalLoss: Focal Loss chuyên dụng cho nhãn Gaussian Heatmap (CenterNet style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross Entropy nhận đầu vào là Logits thô (trước khi qua Sigmoid).
    
    Công thức:
    Loss = - [ pos_weight * y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)) ]
    
    Tham số:
        pos_weight (float): Hệ số nhân phạt cho vùng có quả cầu (thường từ 50.0 - 100.0).
        eps (float): Hằng số cực nhỏ tránh lỗi log(0) dẫn tới NaN / Inf.
    """
    def __init__(self, pos_weight: float = 100.0, eps: float = 1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Tensor dự đoán từ mô hình, shape (B, 3, H, W) - chưa qua Sigmoid.
            targets: Tensor nhãn Ground Truth Gaussian Heatmap, shape (B, 3, H, W) trong [0.0, 1.0].
        Returns:
            Giá trị loss trung bình (scalar).
        """
        # Chuyển logits sang xác suất trong khoảng [eps, 1 - eps] để an toàn về mặt số học
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # Tính Cross Entropy có trọng số
        # Vùng có quả cầu (targets > 0) được nhân thêm hệ số pos_weight
        loss = - (
            self.pos_weight * targets * torch.log(probs) +
            (1.0 - targets) * torch.log(1.0 - probs)
        )

        return loss.mean()


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss (CenterNet / CornerNet style) thiết kế chuyên biệt cho Gaussian Heatmaps.
    
    Ý nghĩa:
    - Tại tâm quả cầu (targets = 1.0): Phạt nặng nếu mô hình thiếu tự tin: (1 - p)^alpha * log(p).
    - Tại vùng xung quanh quả cầu (0 < targets < 1.0): Giảm nhẹ mức phạt nhờ hệ số (1 - targets)^beta.
    - Tại vùng phông nền (targets = 0): Nếu mô hình dự đoán đúng p ~ 0, loss gần như bị triệt tiêu hoàn toàn.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # Tách điểm tâm quả cầu (targets >= 0.99) và các điểm còn lại
        pos_mask = targets.ge(0.99)
        neg_mask = targets.lt(0.99)

        # 1. Loss tại tâm quả cầu
        pos_loss = - torch.pow(1.0 - probs, self.alpha) * torch.log(probs) * pos_mask.float()

        # 2. Loss tại vùng biên và phông nền
        neg_loss = - torch.pow(1.0 - targets, self.beta) * torch.pow(probs, self.alpha) * torch.log(1.0 - probs) * neg_mask.float()

        num_pos = pos_mask.float().sum()
        if num_pos == 0:
            # Nếu toàn bộ frame không có quả cầu nào
            return neg_loss.sum()

        # Chuẩn hóa tổng loss theo số lượng điểm quả cầu xuất hiện
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


def get_loss_fn(loss_name: str = "weighted_bce", pos_weight: float = 100.0) -> nn.Module:
    """
    Hàm khởi tạo Loss Function tiện lợi.
    """
    name = loss_name.lower().strip()
    if name == "weighted_bce":
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == "focal":
        return ModifiedFocalLoss()
    else:
        raise ValueError(f"Không hỗ trợ loss '{loss_name}'. Hãy chọn 'weighted_bce' hoặc 'focal'.")


if __name__ == "__main__":
    print("Testing Loss Functions...")
    # Giả lập Batch B=2, 3 frames, H=288, W=512
    B, C, H, W = 2, 3, 288, 512

    # 1. Giả lập nhãn Ground Truth: Hầu hết là 0, chỉ có 1 đốm sáng nhỏ
    gt = torch.zeros(B, C, H, W)
    gt[:, :, 140:146, 250:256] = 1.0  # Một vùng nhỏ 6x6 pixel là quả cầu

    # 2. Giả lập mô hình dự đoán:
    # Mô hình "lười", đoán toàn số âm (tương đương xác suất ~ 0 sau sigmoid)
    lazy_logits = torch.full((B, C, H, W), -5.0)

    # Mô hình đoán đúng vị trí quả cầu
    good_logits = torch.full((B, C, H, W), -5.0)
    good_logits[:, :, 140:146, 250:256] = 5.0  # Vùng quả cầu có giá trị dương cao

    # So sánh:
    loss_unweighted = nn.BCEWithLogitsLoss()
    loss_weighted = WeightedBCEWithLogitsLoss(pos_weight=100.0)

    print(f"\n--- So sánh BCE thường vs Weighted BCE ---")
    print(f"Kịch bản A (Mô hình lười đoán toàn 0):")
    print(f"  + BCE thường:    {loss_unweighted(lazy_logits, gt).item():.6f} (Rất nhỏ, khiến mô hình tưởng mình giỏi!)")
    print(f"  + Weighted BCE:  {loss_weighted(lazy_logits, gt).item():.6f}  (Bị phạt rất nặng vì bỏ sót quả cầu!)")

    print(f"\nKịch bản B (Mô hình tìm đúng vị trí quả cầu):")
    print(f"  + Weighted BCE:  {loss_weighted(good_logits, gt).item():.6f}  (Loss giảm mạnh khi tìm đúng quả cầu)")
    print("\nKiểm tra Loss Functions thành công!")
