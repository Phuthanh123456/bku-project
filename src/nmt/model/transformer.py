"""TASK 09 — Ghép Encoder-Decoder thành mô hình.  Người làm: Quân.  [Model Core • Bắt buộc]

Xong khi: gradient chảy tới mọi tham số, không có giá trị NaN (bài test 5).

BA ĐIỂM BẮT BUỘC NHỚ:

1. Với Pre-Norm thì PHẢI có một phép chuẩn hóa cuối cùng NGAY TRƯỚC lớp Linear
   xuất ra từ vựng. Luồng residual không bao giờ được chuẩn hóa nên giá trị
   phình dần qua từng lớp. Quên bước này KHÔNG làm chương trình báo lỗi, chỉ
   khiến mô hình mất ổn định mà không rõ nguyên nhân. Bài test 10 bắt lỗi này.

2. RoPE chỉ áp cho self-attention của encoder và của decoder.
   KHÔNG áp cho cross-attention. Bài test 11 bắt lỗi này.

3. Encoder, decoder và lớp xuất dùng CHUNG một ma trận embedding.
   Khởi tạo trọng số theo Xavier.

API SẴN CÓ — gọi đúng tên này, đừng viết lại:

    từ nmt.model.attention      MultiHeadAttention          (TASK 05, Quân)
    từ nmt.model.positional     RoPE, MaHoaViTriSinCos      (TASK 06, Quân)
    từ nmt.model.normalization  build_norm(kieu, d_model, eps)
                                ResidualConnection(d_model, kieu_chuan_hoa,
                                                   vi_tri_chuan_hoa, dropout, eps)
                                build_final_norm(kieu, d_model, vi_tri_chuan_hoa,
                                                 co_chuan_hoa_cuoi, eps)   (TASK 07, Bảo)
    từ nmt.model.layers         build_ffn(kieu, d_model, d_ff, dropout)
                                EncoderLayer, DecoderLayer                 (TASK 08, Bảo)

ResidualConnection đã xử lý sẵn CẢ Pre-Norm lẫn Post-Norm, nên trong lớp encoder
và decoder KHÔNG cần viết nhánh if nào cho `vi_tri_chuan_hoa`. Mỗi khối con cần
MỘT ResidualConnection RIÊNG — dùng chung một cái là hai khối xài chung trọng số
chuẩn hóa, chương trình vẫn chạy nhưng kiến trúc đã sai mà không báo lỗi.

CÒN THIẾU, cần làm trong TASK 09: một hàm chọn mã hóa vị trí theo cấu hình, kiểu

    def build_positional(cfg): ...   # đọc cfg.mo_hinh.ma_hoa_vi_tri

Thiếu hàm này thì tiêu chí "Xong khi" của TASK 06 (đổi cấu hình là chuyển được
giữa RoPE và sin-cos mà không sửa code) chưa chứng minh được, và ablation A4 của
TASK 17 không chạy được.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nmt.model.layers import DecoderLayer, EncoderLayer
from nmt.model.normalization import build_final_norm
from nmt.model.positional import MaHoaViTriSinCos, RoPE


def build_positional(cfg):
    """Factory đọc `mo_hinh.ma_hoa_vi_tri` từ YAML.

    Trả về CẶP (rope, absolute_pe), luôn có đúng một cái khác None:

        rope        áp bên trong attention, chỉ cho query và key của
                    self-attention. Truyền xuống từng lớp qua tham số `rope`.
        absolute_pe cộng thẳng vào embedding một lần ở ngoài cùng, không đụng
                    gì tới bên trong attention.

    Hai phương án tiêm thông tin vị trí ở hai chỗ hoàn toàn khác nhau nên không
    gộp chung một đối tượng được. Đây là hàm khiến tiêu chí "Xong khi" của
    TASK 06 thành hiện thực: đổi một dòng YAML là chuyển được giữa hai phương
    án mà không sửa dòng code nào, và ablation A4 của TASK 17 chạy được.
    """
    model_cfg = cfg.mo_hinh
    max_len = cfg.du_lieu.do_dai_toi_da

    if model_cfg.ma_hoa_vi_tri == "rope":
        rope = RoPE(
            d_head=model_cfg.d_model // model_cfg.so_head,
            theta=model_cfg.rope_theta,
            do_dai_toi_da=max_len,
        )
        return rope, None

    if model_cfg.ma_hoa_vi_tri == "sinusoidal":
        absolute_pe = MaHoaViTriSinCos(
            d_model=model_cfg.d_model,
            do_dai_toi_da=max_len,
            dropout=model_cfg.dropout,
        )
        return None, absolute_pe

    raise ValueError(
        f"ma_hoa_vi_tri không hợp lệ: {model_cfg.ma_hoa_vi_tri!r}. Sửa khóa "
        "mo_hinh.ma_hoa_vi_tri trong YAML thành rope hoặc sinusoidal."
    )


class TransformerNMT(nn.Module):
    """Transformer Encoder-Decoder tự viết, khoảng 48 triệu tham số.

    Toàn bộ lựa chọn kiến trúc đọc từ cfg.mo_hinh, nên đổi kiến trúc là đổi YAML
    chứ không sửa code. Đây là điều kiện để chạy được 4 thí nghiệm của TASK 17.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.mo_hinh

        self.d_model = model_cfg.d_model
        self.vocab_size = cfg.du_lieu.vocab_size
        self.pad_id = 0

        # ĐIỂM 3: encoder, decoder và lớp xuất dùng CHUNG một ma trận embedding.
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)

        # ĐIỂM 2: RoPE áp bên trong attention nên phải truyền xuống từng lớp;
        # sin-cos cộng thẳng vào embedding nên chỉ dùng ở đây. Đúng một cái khác None.
        self.rope, self.absolute_pe = build_positional(cfg)

        self.encoder_layers = nn.ModuleList(
            EncoderLayer(cfg) for _ in range(model_cfg.so_lop_encoder)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(cfg) for _ in range(model_cfg.so_lop_decoder)
        )

        # ĐIỂM 1: Pre-Norm BẮT BUỘC có chuẩn hóa cuối trước lớp xuất từ vựng.
        # build_final_norm tự trả nn.Identity khi dùng Post-Norm, và tự báo lỗi
        # nếu cấu hình pre mà lại tắt co_chuan_hoa_cuoi.
        self.encoder_final_norm = build_final_norm(
            model_cfg.kieu_chuan_hoa, self.d_model,
            model_cfg.vi_tri_chuan_hoa, model_cfg.co_chuan_hoa_cuoi, model_cfg.norm_eps,
        )
        self.decoder_final_norm = build_final_norm(
            model_cfg.kieu_chuan_hoa, self.d_model,
            model_cfg.vi_tri_chuan_hoa, model_cfg.co_chuan_hoa_cuoi, model_cfg.norm_eps,
        )

        self.dropout = nn.Dropout(model_cfg.dropout)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self._init_weights(model_cfg.khoi_tao)

        # Buộc chung trọng số SAU khi khởi tạo, nếu không thì lượt khởi tạo của
        # lop_xuat sẽ ghi đè lên chính ma trận embedding vừa khởi tạo xong.
        if model_cfg.chia_se_embedding:
            self.output_projection.weight = self.embedding.weight

    def _init_weights(self, init_type: str) -> None:
        """Khởi tạo Xavier cho mọi ma trận từ hai chiều trở lên."""
        if init_type != "xavier":
            raise ValueError(
                f"khoi_tao không hợp lệ: {init_type!r}. Sửa khóa mo_hinh.khoi_tao "
                "trong YAML thành xavier."
            )

        for parameter in self.parameters():
            if parameter.dim() >= 2:
                nn.init.xavier_uniform_(parameter)

        # padding_idx phải là vector 0, mà xavier_uniform_ vừa ghi đè lên nó.
        with torch.no_grad():
            self.embedding.weight[self.pad_id].fill_(0.0)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Tra embedding, nhân sqrt(d_model), cộng sin-cos nếu không dùng RoPE.

        Nhân sqrt(d_model) để độ lớn của embedding ngang với độ lớn của vector
        vị trí sin-cos, đúng như bản 2017 mô tả.
        """
        x = self.embedding(token_ids) * (self.d_model ** 0.5)
        if self.absolute_pe is not None:
            # MaHoaViTriSinCos đã có dropout bên trong nên không cộng thêm lần nữa.
            return self.absolute_pe(x)
        return self.dropout(x)

    def encode(self, src_ids, src_mask):
        """src_ids: (batch, len_src) tới (batch, len_src, d_model)."""
        x = self._embed(src_ids)
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask, rope=self.rope)
        return self.encoder_final_norm(x)

    def decode(self, tgt_ids, bo_nho_encoder, tgt_mask, src_mask):
        """tgt_ids: (batch, len_tgt) tới (batch, len_tgt, d_model)."""
        x = self._embed(tgt_ids)
        for layer in self.decoder_layers:
            x = layer(x, bo_nho_encoder, self_mask=tgt_mask, cross_mask=src_mask, rope=self.rope)
        return self.decoder_final_norm(x)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        """Returns: logits có kích thước (batch, len_tgt, vocab_size)."""
        bo_nho_encoder = self.encode(src_ids, src_mask)
        decoder_output = self.decode(tgt_ids, bo_nho_encoder, tgt_mask, src_mask)
        return self.output_projection(decoder_output)

    def dem_tham_so(self) -> dict[str, int]:
        """Đếm tham số theo từng thành phần, cho bảng kết quả đầu ra của Phase 2.

        Returns:
            {"embedding": ..., "encoder": ..., "decoder": ..., "lop_xuat": ..., "tong": ...}

        Đối chiếu với con số tính tay trên giấy, sai lệch phải dưới 1 phần trăm.
        Bảng này chứng minh nhóm hiểu rõ mô hình mình viết ra chứ không chỉ ghép cho chạy.
        """
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Lớp chuẩn hóa cuối tính vào phía nó đứng, vì nó thuộc về chồng lớp đó.
        encoder = count(self.encoder_layers) + count(self.encoder_final_norm)
        decoder = count(self.decoder_layers) + count(self.decoder_final_norm)

        # Khi chia sẻ embedding thì output_projection.weight CHÍNH LÀ
        # embedding.weight, đếm riêng sẽ tính trùng nên ghi 0.
        shared = self.output_projection.weight is self.embedding.weight
        lop_xuat = 0 if shared else count(self.output_projection)

        # Tổng lấy trực tiếp từ tham số của mô hình. PyTorch tự bỏ trùng khi hai
        # thuộc tính trỏ vào cùng một tensor, nên con số này luôn đúng dù có
        # chia sẻ embedding hay không.
        tong = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "embedding": count(self.embedding),
            "encoder": encoder,
            "decoder": decoder,
            "lop_xuat": lop_xuat,
            "tong": tong,
        }
