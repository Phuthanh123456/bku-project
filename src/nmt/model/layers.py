"""TASK 08 — Feed-Forward: SwiGLU + ReLU.  Người làm: Bảo.
TASK 09 — Lớp Encoder / Decoder.         Người làm: Quân.
[Model Core • Bắt buộc]

TASK 08 xong khi: số tham số hai phương án chênh nhau dưới 1% (bài test 9).
    SwiGLU: 3 x 512 x 688  = 1.056.768
    ReLU:   2 x 512 x 1024 = 1.048.576      chênh 0,775 phần trăm

Đặt d_ff sai (giữ 1024 cho SwiGLU) thì mô hình phình thêm khoảng 6,2 triệu
tham số mà không ai nhận ra: 3 x 512 x (1024 trừ 688) x 12 lớp = 6.193.152.

Cả hai khối đều KHÔNG có độ lệch cộng thêm ở các ma trận. Hai lý do:
    1. Giữ số tham số đúng bằng 3 x d_model x d_ff và 2 x d_model x d_ff, nhờ
       vậy phép so của ablation A5 mới công bằng. Xem compare_ffn_parameters.
    2. Ngay trước khối này luôn có một lớp chuẩn hóa nên phần độ lệch gần như
       không đóng góp gì, các mô hình hiện đại đều bỏ.

API công khai của phần TASK 08:

    SwiGLU, FeedForwardReLU   hai phương án feed forward của ablation A5
    build_ffn                 factory đọc khóa `mo_hinh.kieu_ffn`
    count_parameters          đếm tham số học được của một khối
    equivalent_d_ff           tính d_ff của SwiGLU từ d_ff của ReLU
    compare_ffn_parameters    đối chiếu số tham số hai phương án, trả tỉ lệ chênh

Tên ba ma trận (w_gate, w_up, w_down) đặt trùng đúng ký hiệu trong công thức của
bài báo, đọc code tới đâu là dò ra công thức tới đó.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmt.model.attention import MultiHeadAttention
from nmt.model.normalization import ResidualConnection


class SwiGLU(nn.Module):
    """SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Ba ma trận. Nhánh cổng đi qua SiLU rồi nhân TỪNG PHẦN TỬ với nhánh thường,
    sau đó mới qua ma trận thứ ba thu hẹp về d_model.
    SiLU(z) = z * sigmoid(z).  Nguồn: GLU Variants Improve Transformer.

    Vì sao khối này thường hơn ReLU: nhánh cổng học được "cho bao nhiêu phần
    tín hiệu của nhánh thường đi qua" ở TỪNG chiều một, thay vì cắt cụt về 0 như
    ReLU. Đổi lại phải nuôi ba ma trận thay vì hai, nên d_ff bắt buộc hạ xuống
    còn hai phần ba thì phép so mới công bằng.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError(
                f"d_model và d_ff phải là số nguyên dương, nhận được {d_model} và {d_ff}"
            )

        self.d_model = d_model
        self.d_ff = d_ff

        # Ba ma trận, đều không có độ lệch cộng thêm. Lý do ghi ở đầu file.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)    # nhánh cổng
        self.w_up = nn.Linear(d_model, d_ff, bias=False)      # nhánh thường
        self.w_down = nn.Linear(d_ff, d_model, bias=False)    # thu về lại d_model

        # Dropout đặt trên lớp ẩn, ĐÚNG VỊ TRÍ với khối ReLU bên dưới, để hai
        # phương án của ablation A5 chỉ khác nhau duy nhất ở công thức.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) tới cùng kích thước."""
        # F.silu(z) chính là z * sigmoid(z). Đây là HÀM KÍCH HOẠT chứ không phải
        # khối kiến trúc nên không vi phạm quy tắc tự viết của đồ án, lại chạy
        # bằng một nhân gộp sẵn nên ổn định hơn khi tính ở fp16.
        gate = F.silu(self.w_gate(x))             # (batch, seq_len, d_ff)
        up = self.w_up(x)                         # (batch, seq_len, d_ff)

        # Nhân TỪNG PHẦN TỬ. Đây là chỗ dễ sai nhất của cả khối: đảo hai nhánh
        # cho nhau thì kích thước vẫn khớp, chương trình vẫn chạy, mô hình vẫn
        # học được, chỉ là đang học một công thức khác. Bài test đối chiếu với
        # NumPy dựng ra để bắt đúng lỗi đó.
        hidden = self.dropout(gate * up)          # (batch, seq_len, d_ff)

        return self.w_down(hidden)                # (batch, seq_len, d_model)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class FeedForwardReLU(nn.Module):
    """Khối hai ma trận + ReLU của bản 2017 — đối chứng cho ablation A5.

    FFN(x) = ReLU(x @ W_up) @ W_down

    Bản gốc 2017 có độ lệch cộng thêm ở cả hai ma trận, nhóm bỏ đi để số tham số
    khớp với SwiGLU. Phần bỏ đi chỉ là 1536 tham số trên tổng hơn một triệu, nhỏ
    hơn nhiều so với ngưỡng 1% mà mentor yêu cầu.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError(
                f"d_model và d_ff phải là số nguyên dương, nhận được {d_model} và {d_ff}"
            )

        self.d_model = d_model
        self.d_ff = d_ff

        # Hai ma trận, cùng cách đặt tên với SwiGLU để đọc checkpoint đỡ nhầm.
        self.w_up = nn.Linear(d_model, d_ff, bias=False)      # mở rộng lên d_ff
        self.w_down = nn.Linear(d_ff, d_model, bias=False)    # thu về lại d_model

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) tới cùng kích thước."""
        hidden = self.dropout(F.relu(self.w_up(x)))   # (batch, seq_len, d_ff)
        return self.w_down(hidden)                    # (batch, seq_len, d_model)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


def build_ffn(ffn_type: str, d_model: int, d_ff: int, dropout: float = 0.0) -> nn.Module:
    """Factory đọc `mo_hinh.kieu_ffn` từ YAML."""
    if ffn_type == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    if ffn_type == "relu":
        return FeedForwardReLU(d_model, d_ff, dropout)
    raise ValueError(
        f"ffn_type không hợp lệ: {ffn_type!r}. Sửa khóa mo_hinh.kieu_ffn "
        "trong YAML thành swiglu hoặc relu."
    )


# ---------------------------------------------------------------------------
# Ba hàm tiện ích cho tiêu chí  của TASK 08 và báo cáo ablation A5
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    """Đếm số tham số học được của một khối bất kỳ."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def equivalent_d_ff(d_ff_relu: int, multiple: int = 16) -> int:
    """Tính d_ff cho SwiGLU sao cho số tham số xấp xỉ bằng phương án ReLU.

    SwiGLU nuôi ba ma trận còn ReLU chỉ nuôi hai, nên muốn hai bên bằng nhau thì
    d_ff của SwiGLU phải bằng hai phần ba d_ff của ReLU. Giá trị lý tưởng đó
    thường là số lẻ, mà kích thước ma trận nên là bội số của 16 mới khớp khối
    tính của GPU, nên phải làm tròn.

    Trong hai bội số kề nhau thì chọn bên nào cho SỐ THAM SỐ gần phương án ReLU
    hơn, chứ không mặc định làm tròn lên. Làm tròn lên thì với d_ff nhỏ sai số
    vọt quá ngưỡng 1% của mentor, ví dụ 512 làm tròn lên ra 352 và chênh 3%.

        equivalent_d_ff(1024) = 688     vì 1024 x 2/3 = 682.67, chọn 688 thay 672

    Đây chính là chỗ con số 688 trong configs/base.yaml sinh ra, ghi bằng code
    để ba tháng sau đọc lại báo cáo vẫn tra được nó từ đâu mà có.

    LƯU Ý: với d_ff_relu nhỏ thì bước nhảy 16 quá thô, không bội số nào đạt được
    ngưỡng 1%, ví dụ 512 chỉ xuống được tới 1,56%. Gặp trường hợp đó thì hạ
    multiple xuống 8. Luôn kiểm lại bằng compare_ffn_parameters thay vì tin sẵn.
    """
    if d_ff_relu <= 0 or multiple <= 0:
        raise ValueError(
            f"d_ff_relu và multiple phải là số nguyên dương, "
            f"nhận được {d_ff_relu} và {multiple}"
        )

    ideal = d_ff_relu * 2 / 3
    lower = max(multiple, math.floor(ideal / multiple) * multiple)
    upper = lower + multiple

    # So số tham số của hai ứng viên với số tham số của phương án ReLU, giữ bên
    # nào lệch ít hơn. Bỏ d_model đi vì nó là thừa số chung của cả hai bên.
    if abs(3 * lower - 2 * d_ff_relu) <= abs(3 * upper - 2 * d_ff_relu):
        return lower
    return upper


def compare_ffn_parameters(
    d_model: int, d_ff_swiglu: int, d_ff_relu: int
) -> dict[str, float]:
    """Đối chiếu số tham số hai phương án FFN, trả về tỉ lệ chênh lệch.

    Dựng khối thật rồi mới đếm chứ không tính bằng công thức, để hàm này bắt
    được cả lỗi lỡ tay bật độ lệch cộng thêm ở một trong hai khối.

    Trả về dict gồm số tham số mỗi bên và tỉ lệ chênh tính trên bên lớn hơn.
    Yêu cầu của mentor là dưới 1%, quá ngưỡng đó thì ablation A5 đang so hai mô
    hình khác cỡ và kết quả mất ý nghĩa.

        >>> compare_ffn_parameters(512, 688, 1024)["deviation"] < 0.01
        True
    """
    swiglu_params = count_parameters(SwiGLU(d_model, d_ff_swiglu))
    relu_params = count_parameters(FeedForwardReLU(d_model, d_ff_relu))
    deviation = abs(swiglu_params - relu_params) / max(swiglu_params, relu_params)

    return {
        "swiglu": swiglu_params,
        "relu": relu_params,
        "deviation": deviation,
    }


# ===========================================================================
# HẾT PHẦN TASK 08. Từ đây trở xuống là TASK 09, người làm: Quân.
#
# Hai lớp bên dưới mới có khung, chưa cài gì. Quân viết tiếp bắt đầu từ đúng chỗ
# này. TASK 07 và TASK 08 bàn giao sẵn ba thứ để ghép vào, không phải viết lại:
#
#   ResidualConnection   ở nmt.model.normalization, lo luôn cả Pre-Norm lẫn
#                        Post-Norm, chỉ cần truyền cfg.mo_hinh.vi_tri_chuan_hoa
#   build_ffn            ngay phía trên, chọn SwiGLU hay ReLU theo
#                        cfg.mo_hinh.kieu_ffn
#   build_final_norm     lớp chuẩn hóa cuối, gọi ở transformer.py chứ KHÔNG gọi
#                        trong file này
#
# Hai chỗ dễ sai:
#   1. self_attn trả về CẶP (đầu ra, trọng số attention) nên phải lấy [0].
#      ResidualConnection chỉ nhận hàm trả về đúng một tensor.
#   2. Mỗi khối con cần MỘT ResidualConnection RIÊNG. Dùng chung một cái là hai
#      khối xài chung trọng số chuẩn hóa: chương trình vẫn chạy, mô hình vẫn
#      học, nhưng kiến trúc đã sai mà không có gì báo lỗi.
# ===========================================================================


def _build_residual(model_cfg) -> ResidualConnection:
    """Dựng một ResidualConnection từ cfg.mo_hinh.

    Gom lại một chỗ vì mỗi lớp encoder cần hai cái và mỗi lớp decoder cần ba,
    tất cả đều dùng chung bộ tham số này. Vẫn tạo mới mỗi lần gọi, KHÔNG dùng
    lại đối tượng cũ — xem bẫy số 2 ở khối chú thích phía trên.
    """
    return ResidualConnection(
        d_model=model_cfg.d_model,
        norm_type=model_cfg.kieu_chuan_hoa,
        norm_position=model_cfg.vi_tri_chuan_hoa,
        dropout=model_cfg.dropout,
        eps=model_cfg.norm_eps,
    )


class EncoderLayer(nn.Module):
    """Một lớp encoder: self-attention (có RoPE) rồi tới feed forward.

    Pre-Norm : x = x + SubLayer(Norm(x))      mặc định, bản hiện đại
    Post-Norm: x = Norm(x + SubLayer(x))      đối chứng A6, bản 2017

    Hai kiểu chỉ khác nhau ở CHỖ ĐẶT phép chuẩn hóa, không khác gì nữa, và
    ResidualConnection đã xử lý sẵn cả hai nên lớp này không cần nhánh if nào.
    Chọn kiểu bằng `mo_hinh.vi_tri_chuan_hoa`.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.mo_hinh

        self.self_attn = MultiHeadAttention(
            model_cfg.d_model, model_cfg.so_head, model_cfg.dropout
        )
        self.ffn = build_ffn(
            model_cfg.kieu_ffn, model_cfg.d_model, model_cfg.d_ff, model_cfg.dropout
        )

        # BẪY SỐ 2: mỗi khối con MỘT ResidualConnection riêng. Dùng chung một cái
        # là hai khối xài chung trọng số chuẩn hóa, chương trình vẫn chạy nhưng
        # kiến trúc đã sai mà không có gì báo lỗi.
        self.attn_residual = _build_residual(model_cfg)
        self.ffn_residual = _build_residual(model_cfg)

    def forward(self, x, mask=None, rope=None):
        """x: (batch, seq_len, d_model) tới cùng kích thước."""
        # BẪY SỐ 1: self_attn trả về CẶP (đầu ra, trọng số) nên phải lấy [0],
        # vì ResidualConnection chỉ nhận hàm trả về đúng một tensor.
        x = self.attn_residual(x, lambda h: self.self_attn(h, h, h, mask, rope)[0])
        x = self.ffn_residual(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """Một lớp decoder, ba khối con theo thứ tự:

    1. masked self-attention  — CÓ áp RoPE
    2. cross-attention        — KHÔNG áp RoPE
    3. feed forward

    Ba khối con nghĩa là cần BA ResidualConnection riêng. Ở khối cross-attention
    thì query lấy từ x còn key và value lấy từ encoder_memory, và rope truyền
    vào là None một cách có chủ đích chứ không phải quên, xem bài kiểm tra số 11.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.mo_hinh

        self.self_attn = MultiHeadAttention(
            model_cfg.d_model, model_cfg.so_head, model_cfg.dropout
        )
        self.cross_attn = MultiHeadAttention(
            model_cfg.d_model, model_cfg.so_head, model_cfg.dropout
        )
        self.ffn = build_ffn(
            model_cfg.kieu_ffn, model_cfg.d_model, model_cfg.d_ff, model_cfg.dropout
        )

        # Ba khối con nên cần BA ResidualConnection riêng.
        self.self_attn_residual = _build_residual(model_cfg)
        self.cross_attn_residual = _build_residual(model_cfg)
        self.ffn_residual = _build_residual(model_cfg)

    def forward(self, x, encoder_memory, self_mask, cross_mask=None, rope=None):
        """x: (batch, len_tgt, d_model) tới cùng kích thước."""
        # 1. masked self-attention — CÓ áp RoPE
        x = self.self_attn_residual(x, lambda h: self.self_attn(h, h, h, self_mask, rope)[0])

        # 2. cross-attention — query lấy từ x, key và value lấy từ encoder_memory.
        #    rope=None là CÓ CHỦ ĐÍCH chứ không phải quên: query nằm ở câu tiếng
        #    Việt còn key nằm ở câu tiếng Anh, khoảng cách giữa chúng vô nghĩa.
        #    Xem bài kiểm tra số 11.
        x = self.cross_attn_residual(
            x,
            lambda h: self.cross_attn(h, encoder_memory, encoder_memory, cross_mask, None)[0],
        )

        # 3. feed forward
        x = self.ffn_residual(x, self.ffn)
        return x
