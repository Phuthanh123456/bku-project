"""ENVI-NMT — English-Vietnamese Neural Machine Translation.

Transformer hiện đại tự viết từ đầu bằng PyTorch thuần.

QUY TẮC BẤT DI BẤT DỊCH cho toàn bộ package này:
    Trong src/ chỉ được dùng nn.Linear, nn.Embedding, nn.Dropout, nn.Parameter
    và các phép tensor cơ bản (F.softmax, F.silu, F.pad, torch.matmul...).

    KHÔNG dùng: nn.Transformer, nn.TransformerEncoder, nn.MultiheadAttention,
                F.scaled_dot_product_attention, nn.LayerNorm, nn.RMSNorm.

    Các lớp tham chiếu của PyTorch CHỈ được xuất hiện trong tests/ để đối chiếu số.
"""

__version__ = "0.1.0"
