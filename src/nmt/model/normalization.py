"""TASK 07 — Chuẩn hóa: RMSNorm + LayerNorm.  Người làm: Quân.  [Model Core • Bắt buộc]

Xong khi: đối chiếu RMSNorm tự viết với torch.nn.RMSNorm sai lệch dưới 1e-5 (bài test 6).

Trả lời trực tiếp nhận xét 3 của mentor: KHÔNG mặc định RMSNorm tốt hơn
LayerNorm, phải cài cả hai rồi để ablation A1 quyết định bằng số liệu.
A1 là thí nghiệm ưu tiên cao nhất vì mentor nêu trực tiếp.

CÁI BẪY fp16 SỐ 3:
    Phép tính trung bình bình phương BẮT BUỘC thực hiện ở float32 rồi mới ép
    kết quả về dtype ban đầu, giống cách các mô hình mã nguồn mở đang làm.
    Ở fp16 giá trị bình phương dễ vượt 65504 và tràn thành vô cùng, kéo theo
    NaN cho cả mô hình.

RMSNorm khác LayerNorm ở hai điểm: KHÔNG trừ trung bình, và KHÔNG có độ lệch
cộng thêm. Bài test 8 bắt đúng lỗi cài nhầm thành LayerNorm.

API công khai của file này:

    RMSNorm, LayerNorm      hai phương án chuẩn hóa của ablation A1
    build_norm              factory đọc khóa `mo_hinh.kieu_chuan_hoa`
    ResidualConnection      đường phần dư, đặt chuẩn hóa theo Pre hoặc Post (A6)
    build_final_norm        lớp chuẩn hóa cuối bắt buộc có khi dùng Pre-Norm

Nhờ ResidualConnection mà TASK 09 chỉ cần khai báo các khối con, không phải viết
lại hai nhánh Pre-Norm và Post-Norm ở từng lớp encoder và decoder.

QUY ƯỚC ĐẶT TÊN từ TASK 07 trở đi: mọi định danh trong mã nguồn viết bằng tiếng
Anh theo PEP 8, chú thích và thông báo lỗi viết bằng tiếng Việt. Khóa trong file
YAML vẫn giữ tiếng Việt vì đó là dữ liệu cấu hình chứ không phải mã nguồn, nên
thông báo lỗi luôn nhắc kèm tên khóa YAML tương ứng để người sửa biết sửa ở đâu.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight — https://arxiv.org/abs/1910.07467

    Khoảng 10 dòng, đây là thành phần dễ nhất trong bốn thành phần mới.
    Nhớ hằng số epsilon chống chia cho 0.

    Ý tưởng: chỉ kéo mọi vector về cùng một ĐỘ LỚN, không quan tâm tới trung
    bình của chúng. Rẻ hơn LayerNorm vì bỏ được một lượt duyệt tính trung bình
    và bỏ luôn phép cộng độ lệch.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model phải là số nguyên dương, nhận được {d_model}")

        self.d_model = d_model
        self.eps = eps

        # Chỉ có DUY NHẤT hệ số nhân, khởi tạo bằng 1 để lúc mới bắt đầu lớp này
        # không làm méo tín hiệu. Không có độ lệch cộng thêm, đây là điểm khác
        # thứ hai so với LayerNorm.
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) tới (..., d_model). Chuẩn hóa trên CHIỀU CUỐI CÙNG."""
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Chiều cuối của đầu vào là {x.size(-1)} nhưng lớp chuẩn hóa "
                f"được tạo cho d_model = {self.d_model}"
            )

        input_dtype = x.dtype

        # CÁI BẪY fp16 SỐ 3. Bình phương rồi lấy trung bình ở float32.
        # Ở fp16 số lớn nhất biểu diễn được chỉ là 65504, nên chỉ cần giá trị đầu
        # vào cỡ 300 là bình phương đã tràn thành vô cùng, rsqrt(vô cùng) ra 0 và
        # toàn bộ đầu ra thành 0. Chương trình KHÔNG báo lỗi gì cả, chỉ là mô hình
        # ngừng học. Bài test 12 và bài test fp16 của TASK 07 bắt đúng lỗi này.
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)   # (..., 1)

        # rsqrt là nghịch đảo căn bậc hai, nhanh hơn chia cho sqrt một nhịp.
        # Cộng eps TRƯỚC khi lấy căn để không bao giờ chia cho 0 khi cả vector bằng 0.
        normalized = x_fp32 * torch.rsqrt(mean_square + self.eps)

        # Ép về đúng dtype đầu vào rồi mới nhân hệ số, giữ đúng giao kèo
        # "vào kiểu số nào thì ra kiểu số đó" để lớp phía sau không bị đổi kiểu
        # ngoài ý muốn khi chạy trong autocast fp16.
        return normalized.to(input_dtype) * self.weight.to(input_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


class LayerNorm(nn.Module):
    """LayerNorm tự viết — phương án đối chứng cho ablation A1.

    Trừ trung bình, chia độ lệch chuẩn, có cả hệ số nhân và độ lệch cộng thêm.
    Tự viết chứ không gọi lớp dựng sẵn của PyTorch, vì quy tắc của đồ án là
    không dùng lớp dựng sẵn trong src/.

    Công thức: (x - mean) / sqrt(var + eps) * weight + bias

    Phương sai tính theo kiểu chia cho n chứ không chia cho n trừ 1, đúng như
    cách PyTorch làm. Chia nhầm mẫu số thì sai lệch khi đối chiếu số vượt 1e-5
    ngay, mà mô hình vẫn train được nên rất khó nhận ra nếu không có bài test.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model phải là số nguyên dương, nhận được {d_model}")

        self.d_model = d_model
        self.eps = eps

        # Hai tham số học được, gấp đôi RMSNorm. Đây chính là phần chi phí thêm
        # mà ablation A1 phải trả lời là có đáng hay không.
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) tới (..., d_model). Chuẩn hóa trên CHIỀU CUỐI CÙNG."""
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Chiều cuối của đầu vào là {x.size(-1)} nhưng lớp chuẩn hóa "
                f"được tạo cho d_model = {self.d_model}"
            )

        input_dtype = x.dtype

        # Cùng cái bẫy fp16 số 3 như RMSNorm. Bình phương độ lệch cũng tràn ở
        # fp16 y hệt, nên toàn bộ phần tính toán nằm ở float32.
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)                  # (..., 1)
        centered = x_fp32 - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)     # (..., 1)
        normalized = (centered * torch.rsqrt(variance + self.eps)).to(input_dtype)

        return normalized * self.weight.to(input_dtype) + self.bias.to(input_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


def build_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    """Factory đọc `mo_hinh.kieu_chuan_hoa` từ YAML.

    Cả mô hình chỉ được tạo lớp chuẩn hóa qua hàm này, nhờ vậy đổi kiến trúc
    chỉ cần sửa một dòng cấu hình.
    """
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps)
    if norm_type == "layernorm":
        return LayerNorm(d_model, eps)
    raise ValueError(
        f"norm_type không hợp lệ: {norm_type!r}. Sửa khóa mo_hinh.kieu_chuan_hoa "
        "trong YAML thành rmsnorm hoặc layernorm."
    )


class ResidualConnection(nn.Module):
    """Đường phần dư kèm chuẩn hóa, đặt được theo Pre-Norm hoặc Post-Norm.

    Pre-Norm : y = x + Dropout(SubLayer(Norm(x)))      mặc định, bản hiện đại
    Post-Norm: y = Norm(x + Dropout(SubLayer(x)))      đối chứng A6, bản 2017

    Hai kiểu chỉ khác nhau ở CHỖ ĐẶT phép chuẩn hóa. Gói chung vào một lớp để
    TASK 09 không phải viết hai nhánh if ở từng lớp encoder và decoder, và để
    ablation A6 đổi được bằng đúng một dòng YAML.

    Vì sao Pre-Norm ổn định hơn: đường phần dư đi thẳng từ đầu tới cuối mà không
    đi qua phép chuẩn hóa nào, nên gradient truyền ngược về không bị co lại qua
    từng lớp. Đổi lại giá trị trên đường đó phình dần, nên BẮT BUỘC phải có một
    lớp chuẩn hóa cuối, xem hàm build_final_norm bên dưới.

    Khối con phải là một hàm nhận một tensor và trả về một tensor CÙNG kích
    thước. Với attention (trả về cặp gồm đầu ra và trọng số) thì bọc lại như sau::

        y = residual(x, lambda h: self.self_attn(h, h, h, mask, rope)[0])

    Ví dụ khai báo trong TASK 09::

        self.attn_residual = ResidualConnection(
            d_model=cfg.mo_hinh.d_model,
            norm_type=cfg.mo_hinh.kieu_chuan_hoa,
            norm_position=cfg.mo_hinh.vi_tri_chuan_hoa,
            dropout=cfg.mo_hinh.dropout,
            eps=cfg.mo_hinh.norm_eps,
        )
    """

    def __init__(
        self,
        d_model: int,
        norm_type: str = "rmsnorm",
        norm_position: str = "pre",
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if norm_position not in ("pre", "post"):
            raise ValueError(
                f"norm_position không hợp lệ: {norm_position!r}. Sửa khóa "
                "mo_hinh.vi_tri_chuan_hoa trong YAML thành pre hoặc post."
            )

        self.norm_position = norm_position
        self.norm = build_norm(norm_type, d_model, eps)

        # Dropout đặt trên ĐẦU RA của khối con, trước khi cộng vào đường phần dư,
        # đúng như bản 2017 mô tả. Không đặt trên chính đường phần dư, làm vậy là
        # cắt ngẫu nhiên tín hiệu gốc và mô hình rất khó học.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """x: (batch, seq_len, d_model) tới cùng kích thước."""
        if self.norm_position == "pre":
            # Chuẩn hóa TRƯỚC khi vào khối con. Đường phần dư giữ nguyên x.
            return x + self.dropout(sublayer(self.norm(x)))

        # Post-Norm: cộng phần dư xong mới chuẩn hóa, nên đường phần dư bị phép
        # chuẩn hóa cắt ngang ở mỗi lớp. Đây là lý do Post-Norm gần như bắt buộc
        # phải có warmup, thiếu warmup thì loss phân kỳ ngay những bước đầu.
        return self.norm(x + self.dropout(sublayer(x)))

    def extra_repr(self) -> str:
        return f"norm_position={self.norm_position}"


def build_final_norm(
    norm_type: str,
    d_model: int,
    norm_position: str,
    use_final_norm: bool = True,
    eps: float = 1e-6,
) -> nn.Module:
    """Lớp chuẩn hóa cuối, đặt ngay trước lớp Linear xuất ra từ vựng.

    Với Pre-Norm thì giá trị trên đường phần dư phình dần qua từng lớp vì không
    có chỗ nào kéo nó về lại. Thiếu lớp chuẩn hóa cuối KHÔNG làm chương trình
    báo lỗi, chỉ khiến mô hình mất ổn định lúc huấn luyện mà không rõ nguyên
    nhân. Đây là lỗi rất phổ biến, bài kiểm tra số 10 dựng ra để bắt đúng nó.

    Với Post-Norm thì lớp cuối cùng của chồng đã tự chuẩn hóa rồi nên trả về
    nn.Identity, tức không làm gì thêm.
    """
    if norm_position not in ("pre", "post"):
        raise ValueError(
            f"norm_position không hợp lệ: {norm_position!r}. Sửa khóa "
            "mo_hinh.vi_tri_chuan_hoa trong YAML thành pre hoặc post."
        )

    if norm_position == "pre" and not use_final_norm:
        raise ValueError(
            "vi_tri_chuan_hoa = 'pre' thì BẮT BUỘC co_chuan_hoa_cuoi = true. "
            "Xem bài kiểm tra số 10 trong tests/test_model_correctness.py."
        )

    if use_final_norm:
        return build_norm(norm_type, d_model, eps)
    return nn.Identity()
