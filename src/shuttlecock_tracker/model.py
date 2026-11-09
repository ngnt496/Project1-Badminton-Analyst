"""
Nhận diện cầu từ 3 frames liên tiếp
Encoder-Decoder dạng U-Net & Skip Connections
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpCatTripleConv(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # x2 độ phân giải
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # channels = in_channels + skip_channels
        self.triple_conv = TripleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Nối theo chiều kênh
        return self.triple_conv(x)


class UpCatDoubleConv(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class ShuttlecockTrackerNet(nn.Module):
    """
    Input:  (B, 9, 288, 512) -> 3 frames liên tiếp ghép kênh (3 frame x 3 kênh RGB = 9)
    Output: (B, 3, 288, 512) -> Heatmap xác suất [0, 1] cho từng frame (t-1, t, t+1)
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 1. ENCODER
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = TripleConv(128, 256)
        self.enc4 = TripleConv(256, 512)
        self.bottleneck = TripleConv(512, 512)

        # 2. DECODER (Upsample + Cat + Conv)
    
        self.dec1 = UpCatTripleConv(in_channels=512, skip_channels=512, out_channels=512)
        self.dec2 = UpCatTripleConv(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = UpCatDoubleConv(in_channels=256, skip_channels=128, out_channels=128)
        self.dec4 = UpCatDoubleConv(in_channels=128, skip_channels=64, out_channels=64)
        
        # 3. OUTPUT LAYER (Conv 1x1 + Sigmoid)
        # 64 channels -> 3 heatmaps (t-1, t, t+1) với giá trị xác suất từ 0.0 đến 1.0
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Khởi tạo trọng số tường minh theo chuẩn Kaiming He cho mạng dùng ReLU
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # --- ENCODER ---
        s1 = self.enc1(x)                   # (B, 64, 288, 512)

        x = self.maxpool(s1)                # (B, 64, 144, 256)
        s2 = self.enc2(x)                   # (B, 128, 144, 256)

        x = self.maxpool(s2)                # (B, 128, 72, 128)
        s3 = self.enc3(x)                   # (B, 256, 72, 128)

        x = self.maxpool(s3)                # (B, 256, 36, 64)
        s4 = self.enc4(x)                   # (B, 512, 36, 64)

        x = self.maxpool(s4)                # (B, 512, 18, 32)
        b = self.bottleneck(x)              # (B, 512, 18, 32)

        # --- DECODER ---
        x = self.dec1(b, s4)                # (B, 512, 36, 64)
        x = self.dec2(x, s3)                # (B, 256, 72, 128)
        x = self.dec3(x, s2)                # (B, 128, 144, 256)
        x = self.dec4(x, s1)                # (B, 64, 288, 512)

        # --- HEAD ---
        logits = self.final_conv(x)         # (B, 3, 288, 512)

        if return_logits:
            return logits                   # Trả về logits thô (dùng khi tính BCE loss có pos_weight)
        return self.sigmoid(logits)         # Trả về heatmap xác suất [0, 1]


if __name__ == "__main__":
    print("Testing ShuttlecockTrackerNet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(2, 9, 288, 512, device=device)
    model = ShuttlecockTrackerNet(in_channels=9, out_channels=3).to(device)

    # Đếm số lượng trọng số có thể học được (learnable parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số trọng số (Learnable parameters): {total_params:,}")

    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: min={output.min().item():.4f}, max={output.max().item():.4f}")
    assert output.shape == (2, 3, 288, 512), "Kích thước đầu ra không khớp!"
    print("Xác thực kiến trúc và trọng số mô hình thành công!")
