import torch
import torch.nn as nn
import torch.nn.functional as F

""""
Bỏ kết nối tắt trong MBconv, kết nối tắt bên ngoài xử lý và droppath
khối DWC7x7 theo ConvMixer.
Window attention có sử dụng Relative Position Bias
Giai đoạn 4 sủ dụng Global attention
"""
# ==========================================
# 1. CÁC MODULE PHỤ TRỢ CƠ BẢN
# ==========================================

class DropPath(nn.Module):
    """ Stochastic Depth: Loại bỏ ngẫu nhiên các nhánh trong mạng sâu """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    """ Áp dụng LayerNorm cho tensor ảnh [B, C, H, W] """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        return x

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation (SE) Block """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Mlp(nn.Module):
    """ Multi-Layer Perceptron (MLP) sử dụng nn.Linear """
    def __init__(self, in_features, mlp_ratio=4.0, drop=0.):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Chuyển (B, C, H, W) sang (B, H, W, C) để tương thích nn.Linear
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # Chuyển ngược lại (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
#-----------------------------------------------------------
class MBConv(nn.Module):
    """ Mobile Inverted Bottleneck (Có BatchNorm nội bộ) """
    def __init__(self, in_channels, expansion=2):
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim) # Bổ sung BN
        self.act1 = nn.GELU()
        
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim) # Bổ sung BN
        self.act2 = nn.GELU()
        
        self.se = SEBlock(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, in_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels) # Bổ sung BN
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.dwconv(x)))
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        return x

# ==========================================
# 2. KHỐI TÍNH TOÁN CỐT LÕI (BLOCKS)
# ==========================================
class DWDCBlock(nn.Module):
    """ Giai đoạn 1 & 2: Nhánh trộn không gian nối tiếp (5x5 -> 7x7) với 1 kết nối tắt chung """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # --- 1. Nhánh MBConv ---
        self.norm1 = nn.BatchNorm2d(dim)
        self.mbconv = MBConv(dim)
        
        # --- 2. Nhánh Spatial Mixing (Chuỗi nối tiếp 5x5 -> 7x7) ---
        self.norm2 = nn.BatchNorm2d(dim) # Pre-Norm cho toàn bộ nhánh
        
        self.dwconv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.bn5x5 = nn.BatchNorm2d(dim)
        self.act5x5 = nn.GELU()
        
        self.dwconv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=6, dilation=2, groups=dim, bias=False)
        
        # --- 3. Nhánh Channel Mixing ---
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act_pw = nn.GELU()
        self.norm3 = nn.BatchNorm2d(dim)
        
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # 1. Tinh chỉnh cục bộ (MBConv)
        x = x + self.drop_path(self.mbconv(self.norm1(x)))
        
        # 2. Xử lý không gian đa quy mô (Dùng chung 1 kết nối tắt)
        res = x
        x = self.norm2(x)             # Chuẩn hóa đầu vào của nhánh
        x = self.dwconv5x5(x)         # Trích xuất đặc trưng 5x5
        x = self.bn5x5(x)             # Chuẩn hóa trung gian
        x = self.act5x5(x)            # Kích hoạt phi tuyến trung gian
        x = self.dwconv7x7(x)         # Mở rộng vùng cảm nhận bằng 7x7 (dilation=2)
        x = res + self.drop_path(x)   # Cộng kết nối tắt chung cho toàn nhánh
        
        # 3. Trộn kênh (Conv1x1)
        x = self.pwconv(x)
        x = self.act_pw(x)
        x = self.norm3(x)
        
        return x
#------------------------------------
# class DWDCBlock(nn.Module):
    # """ Giai đoạn 1 & 2: Kiến trúc theo phong cách ConvMixer """
    # def __init__(self, dim, drop_path=0.):
        # super().__init__()
        # # --- Nhánh MBConv ---
        # self.norm1 = nn.BatchNorm2d(dim)
        # self.mbconv = MBConv(dim)
        
        # # --- Nhánh Spatial Mixing (Có kết nối tắt) ---
        # self.norm2 = nn.BatchNorm2d(dim)
        # self.dwconv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=6, dilation=2, groups=dim, bias=False)
        
        # # --- Nhánh Channel Mixing (Không kết nối tắt) ---
        # self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # self.act = nn.GELU()
        # self.norm3 = nn.BatchNorm2d(dim)
        
        # self.drop_path = DropPath(drop_path)

    # def forward(self, x):
        # # 1. MBConv (Cục bộ)
        # x = x + self.drop_path(self.mbconv(self.norm1(x)))
        
        # # 2. DWConv 7x7 (Cộng kết nối tắt tại đây giống ConvMixer)
        # res = x
        # x = self.norm2(x)
        # x = self.dwconv7x7(x)
        # x = res + self.drop_path(x)
        
        # # 3. Pointwise Conv1x1 -> GELU -> BatchNorm (Đi thẳng)
        # x = self.pwconv(x)
        # x = self.act(x)
        # x = self.norm3(x)
        
        # return x
#------------------------------
class WindowAttention(nn.Module):
    """ Window-based Multi-head Self-Attention với Relative Position Bias """
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 1. Bảng tham số học được cho vị trí tương đối
        # Kích thước: (2*window_size - 1) * (2*window_size - 1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # 2. Tạo ma trận chỉ mục (index) cho các vị trí tương đối bên trong cửa sổ
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, W, W]
        coords_flatten = torch.flatten(coords, 1)  # [2, W*W]
        
        # Tính khoảng cách tương đối giữa mỗi cặp pixel
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, W*W, W*W]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [W*W, W*W, 2]
        
        # Dịch chuyển (shift) để không có giá trị âm
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        relative_position_index = relative_coords.sum(-1)  # [W*W, W*W]
        self.register_buffer("relative_position_index", relative_position_index)

        # Sử dụng nn.Linear thay cho Conv2d để dễ xử lý tensor (B, N, C)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window_size
        
        # 1. Chuyển đổi [B, C, H, W] -> Dạng chuỗi sequence cho Transformer [B * num_windows, W*W, C]
        x = x.view(B, C, H // w, w, W // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, w * w, C)
        
        B_, N, C_ = x.shape
        
        # 2. Tính Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. Tính Attention Score
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 4. --- CỘNG RELATIVE POSITION BIAS (Linh hồn của Swin/MaxViT) ---
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            w * w, w * w, -1)  # [W*W, W*W, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, W*W, W*W]
        
        attn = attn + relative_position_bias.unsqueeze(0)
        # ----------------------------------------------------------------
        
        attn = self.softmax(attn)
        
        # 5. Nhân với V và chiếu qua lớp Proj
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C_)
        x = self.proj(x)
        
        # 6. Khôi phục lại định dạng ảnh 2D [B, C, H, W]
        x = x.view(B, H // w, W // w, w, w, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x
#--------------
class MPViTBlock(nn.Module):
    """ Giai đoạn 3: Khối Trộn Patch Bất Đối Xứng (2 Cụm Attn+FFN) """
    def __init__(self, dim, num_heads, window_size=7, drop_path=0.):
        super().__init__()
        self.shift_h = 4
        self.shift_w = 3
        
        # 1. Cửa ngõ: MBConv
        self.norm1 = LayerNorm2d(dim)
        self.mbconv = MBConv(dim)
        
        # --- CỤM 1: Cục bộ ---
        self.norm2 = LayerNorm2d(dim)
        self.attn1 = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm3 = LayerNorm2d(dim)
        self.ffn1 = Mlp(in_features=dim, mlp_ratio=4.0)
        
        # --- CỤM 2: Xuyên cửa sổ (Roll) ---
        self.norm4 = LayerNorm2d(dim)
        self.attn2 = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm5 = LayerNorm2d(dim)
        self.ffn2 = Mlp(in_features=dim, mlp_ratio=4.0)
        
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # 1. Tinh chỉnh không gian cục bộ đầu vào (MBConv)
        x = x + self.drop_path(self.mbconv(self.norm1(x)))
        
        # ==========================================
        # CỤM 1: Tương tác cục bộ trong cửa sổ gốc
        # ==========================================
        x = x + self.drop_path(self.attn1(self.norm2(x)))
        x = x + self.drop_path(self.ffn1(self.norm3(x)))
        
        # ==========================================
        # CỤM 2: Mixing Patches (Roll bất đối xứng)
        # ==========================================
        res = x
        x_shift = self.norm4(x)
        
        # Cuộn 4x3 tạo cửa sổ lai
        x_mixed = torch.roll(x_shift, shifts=(self.shift_h, self.shift_w), dims=(2, 3))
        
        # Tính Attention trên cửa sổ lai
        x_attn2 = self.attn2(x_mixed)
        
        # Cuộn ngược khôi phục không gian gốc
        x_restored = torch.roll(x_attn2, shifts=(-self.shift_h, -self.shift_w), dims=(2, 3))
        x = res + self.drop_path(x_restored)
        
        # FFN kết thúc cụm 2
        x = x + self.drop_path(self.ffn2(self.norm5(x)))
        
        return x
#----------------------------------
class Stage4HybridBlock(nn.Module):
    """ Giai đoạn 4: Khối Hybrid kết hợp MBConv và Global Attention (Thiết kế Tuần tự) """
    def __init__(self, dim, num_heads, window_size=7, drop_path=0.):
        super().__init__()
        
        # 1. Cửa ngõ tinh chỉnh cục bộ (MBConv)
        self.norm1 = LayerNorm2d(dim)
        self.mbconv = MBConv(dim)
        
        # 2. Cụm Global Attention (Không Mixing/Roll)
        self.norm2 = LayerNorm2d(dim)
        # Hoạt động như Global Attention vì feature map size == window_size == 7
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        
        # 3. Cụm FFN
        self.norm3 = LayerNorm2d(dim)
        self.ffn = Mlp(in_features=dim, mlp_ratio=4.0)
        
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # 1. Nhánh MBConv (Đã bọc kết nối tắt và DropPath an toàn bên ngoài)
        x = x + self.drop_path(self.mbconv(self.norm1(x)))
        
        # 2. Nhánh Global Attention
        x = x + self.drop_path(self.attn(self.norm2(x)))
        
        # 3. Mạng nơ-ron tiến (FFN)
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x

# ==========================================
# 3. KIẾN TRÚC TỔNG THỂ (HYBRID MODEL)
# ==========================================

class SEROLLMPViTGA(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, 
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 5, 2], 
                 num_heads_s3=8, num_heads_s4=16, drop_path_rate=0.1):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # S0: Khối LViT Down Sampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            SEBlock(embed_dims[0])
        )
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                if i < 2:
                    # Giai đoạn 1 & 2 (56x56, 28x28): DWDCBlock
                    blocks.append(DWDCBlock(dim=embed_dims[i], drop_path=dpr[cur]))
                elif i == 2:
                    # Giai đoạn 3 (14x14): MPViTBlock (Mixing)
                    blocks.append(MPViTBlock(dim=embed_dims[i], num_heads=num_heads_s3, drop_path=dpr[cur]))
                else:
                    # Giai đoạn 4 (7x7): Stage4HybridBlock (Global, no Mixing)
                    blocks.append(Stage4HybridBlock(dim=embed_dims[i], num_heads=num_heads_s4, drop_path=dpr[cur]))
                cur += 1
            self.stages.append(blocks)
            
            # Khối hạ độ phân giải không gian
            if i < 3:
                self.downsamples.append(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1)
                )
        
        # Khối phân loại (Head)
        self.norm = nn.BatchNorm1d(embed_dims[3])
        self.head = nn.Linear(embed_dims[3], num_classes)

    def forward(self, x):
        # 1. Stem
        x = self.stem(x)
        
        # 2. Các giai đoạn
        for i in range(4):
            for block in self.stages[i]:
                x = block(x)
            if i < 3:
                x = self.downsamples[i](x)
                
        # 3. Global Average Pooling & Head
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.norm(x)
        x = self.head(x)
        
        return x

# test model:
# model = HybridSEROLLPViT()
# dummy = torch.randn(2, 3, 224, 224)
# out = model(dummy)
# print(out.shape) # torch.Size([2, 1000])
