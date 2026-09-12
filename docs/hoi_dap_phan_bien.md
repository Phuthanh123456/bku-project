# Hỏi đáp phản biện

Chuẩn bị cho buổi bảo vệ. Mọi con số trong file này lấy từ kết quả đo thật, có
thể tra lại trong `results/` và trên Hugging Face.

**Đọc mục [Bốn chỗ nhóm yếu](#bốn-chỗ-nhóm-yếu-phải-biết-trước) trước tiên.**
Đó là những câu nhóm không có câu trả lời hoàn hảo, và biết trước thì trả lời
trung thực được, còn bị hỏi bất ngờ thì dễ nói bừa.

---

## Bốn chỗ nhóm yếu, phải biết trước

### 1. Câu "chỉ dùng tst2013 một lần" trên slide là SAI

Slide 17 ghi tst2013 *"chỉ dùng một lần ở bước cuối"*. Thực tế tst2013 đã được
chấm cho **cả 6 lượt ablation**, cộng lần so greedy với beam. Nhiều hơn một lần
khá xa.

**Nên sửa câu đó trước khi bảo vệ** thành:

> "tst2013 chỉ dùng để **báo cáo**, không dùng để **chọn** siêu tham số hay
> checkpoint — mọi lựa chọn đều dựa trên tst2012."

Câu mới vừa đúng sự thật vừa giữ được tinh thần ban đầu. Để nguyên câu cũ là tự
đưa cho hội đồng một chỗ bắt bẻ bằng chính tài liệu của nhóm.

### 2. Lượt 3.000 bước tốt hơn lượt 17.000 bước — chưa giải thích được

| Lượt | Số bước | loss dev |
|---|---|---|
| `iwslt_base_v1_seed42` | 17.000 | 2,3866 |
| `abl3000_base_seed42` | 3.000 | **2,2767** |

Vấn đề: lượt 17.000 bước **cũng đi qua mốc 3.000**. Nếu lúc đó nó đạt 2,28 thì
bản tốt nhất của nó đã phải là 2,28 chứ không phải 2,3866.

Nên giữa hai lượt còn khác nhau thứ gì đó ngoài độ dài. Giả thuyết: nhịp đánh
giá khác nhau (500 bước so với 1.000 bước — đánh giá dày hơn thì bắt được đáy
tốt hơn), hoặc file cấu hình nền đã thay đổi giữa hai lượt.

**Chưa kiểm chứng được giả thuyết nào.** Trả lời trung thực:

> "Em ghi nhận chênh lệch này và chưa giải thích được dứt điểm. Em có hai giả
> thuyết nhưng chưa kiểm, nên em không muốn khẳng định."

### 3. Hai seed là mức tối thiểu, không phải mức đủ

Hai seed cho biết **khoảng dao động**, nhưng không đủ để tính khoảng tin cậy hay
giá trị p. Muốn phát biểu thống kê đàng hoàng cần ít nhất 5 seed.

Nhóm dừng ở 2 vì ngân sách GPU. **Nói thẳng đó là giới hạn**, đừng giả vờ là đủ.

### 4. Dropout 0,3 chưa được ablation

0,3 là lựa chọn có lý do (dữ liệu ít nên cần điều hoà mạnh) nhưng **chưa có số
chứng minh**. Bảng khảo sát chỉ đo thời gian của 0,1 / 0,3 / 0,5, không đo BLEU.

### Một chỗ nhỏ dễ bị soi: 24 hay 21 cặp rò rỉ?

Slide 6 ghi "xoá **24 cặp**" nhưng bảng chi tiết ghi "**21** rò rỉ". Không mâu
thuẫn: 24 cặp bị phát hiện trùng, nhưng 3 trong số đó đã bị các bước lọc trước
đó (rỗng, trùng lặp, quá dài) loại mất rồi, nên tới bước chống rò rỉ chỉ còn 21
cặp để xoá. Nếu bị hỏi thì giải thích vậy.

---

## Nhóm 2 — Batch và khả năng tái lập

### Tại sao batch theo 4096 token thay vì theo số câu?

Vì câu dài ngắn rất khác nhau. Batch "32 câu" có thể là 32 câu ngắn 5 token
(160 token) hoặc 32 câu dài 100 token (3.200 token) — chênh nhau **20 lần**.

Hệ quả khi batch theo số câu:

- **Tải GPU giật cục** — batch nhẹ thì GPU nhàn, batch nặng thì suýt tràn bộ nhớ
- **Phải đặt số câu theo trường hợp xấu nhất**, nên trung bình GPU chạy non tải
- **Gradient nhiễu không đều** — batch ít token cho ước lượng gradient kém tin cậy hơn

Batch theo **số token** giữ khối lượng tính toán mỗi bước gần như không đổi. GPU
luôn chạy gần đầy tải, và mỗi bước cập nhật dựa trên lượng dữ liệu tương đương.

**Vì sao 4096?** Đo thật, không đoán:

| Token/batch | Giây/bước | Bước/epoch | Tổng giờ GPU | Khả thi |
|---|---|---|---|---|
| 2.048 | 0,8019 | 351 | 13,37 | có |
| **4.096** | **1,5000** | **177** | **25,00** | **có** |
| 8.192 | — | — | — | **tràn bộ nhớ T4** |

8.192 loại ngay vì tràn VRAM. 4.096 chọn vì cho batch lớn hơn mà vẫn nằm trong
ngân sách 30 giờ GPU mỗi tuần.

Cộng thêm **cộng dồn gradient 4 micro-batch**, batch hiệu dụng là **16.384
token**. Transformer huấn luyện ổn định hơn hẳn với batch lớn — đây là cách có
batch lớn mà không cần GPU lớn.

### Gom câu độ dài gần nhau vào cùng batch giúp gì?

Giúp **bớt ô đệm lãng phí**.

Trong một batch, mọi câu phải dài bằng nhau, nên câu ngắn được đệm `<pad>` cho
bằng câu dài nhất. Ô đệm vẫn được GPU tính toán đầy đủ rồi **vứt bỏ** ở bước
cuối.

Ví dụ cụ thể — một batch 10 câu:

| Cách gom | Độ dài các câu | Ô thật | Ô đệm | Lãng phí |
|---|---|---|---|---|
| Ngẫu nhiên | 1 câu 100, 9 câu 10 | 190 | 810 | **81%** |
| Theo độ dài | 10 câu ~100 | ~1.000 | ~0 | **~0%** |

Cùng một khối lượng tính toán, cách thứ hai xử lý gấp **5 lần** dữ liệu thật.

Điều này đặc biệt quan trọng ở khâu **sinh câu**: decoder chạy tự hồi quy cho
tới khi câu **dài nhất** trong batch xong. Một câu 100 token đứng chung với chín
câu 10 token bắt cả batch chạy đủ 100 bước.

### Padding mask và causal mask khác nhau thế nào? Vì sao cần cả hai?

Hai mặt nạ giải quyết **hai vấn đề hoàn toàn khác nhau**.

**Padding mask** — che ô đệm.

Vấn đề: câu ngắn được đệm `<pad>` cho đủ độ dài batch. Không che thì attention
coi `<pad>` như token thật, và biểu diễn của câu bị pha loãng bởi thứ vô nghĩa.

Hình dạng `[B, 1, 1, S]`, `True` tại vị trí thật. Dùng ở **cả encoder lẫn
decoder**, và cả ở cross-attention (che phía nguồn).

**Causal mask** — che tương lai.

Vấn đề: lúc huấn luyện, decoder được đưa **cả câu đích đúng** vào cùng lúc để
tính song song cho nhanh. Nếu vị trí thứ 5 được nhìn vị trí thứ 6, nó **nhìn
thẳng vào đáp án**. Mô hình sẽ học mẹo "copy từ kế tiếp" thay vì học dịch — loss
train xuống rất đẹp, còn lúc dịch thật thì hỏng hoàn toàn vì không có đáp án để
copy.

Hình dạng tam giác dưới. Chỉ dùng ở **self-attention của decoder**.

| | Che gì | Dùng ở đâu | Thiếu thì sao |
|---|---|---|---|
| Padding | ô `<pad>` | encoder, decoder, cross | biểu diễn bị nhiễu |
| Causal | vị trí tương lai | chỉ self-attention decoder | mô hình gian lận, vô dụng lúc dịch |

Cần **cả hai** vì trong self-attention của decoder, một vị trí vừa phải không
nhìn tương lai, vừa phải không nhìn ô đệm. Hai mặt nạ được gộp lại bằng phép AND.

**Một cái bẫy fp16 ở đây:** giá trị dùng để che phải là `torch.finfo(dtype).min`
chứ không phải `-1e9`. Ở fp16, `-1e9` vượt dải biểu diễn và thành `-inf`; softmax
của một hàng toàn `-inf` cho ra **NaN**, và NaN lan ra cả mô hình.

### Kaggle ngắt giữa epoch, làm sao gặp lại đúng thứ tự batch?

Vấn đề thật không phải "nạp lại trọng số", mà là **thứ tự batch**.

Nếu phiên đứt giữa epoch 17 sau khi đã tiêu thụ 93 batch, mà lượt sau bắt đầu
lại từ batch đầu của epoch 17, thì 93 batch đó được học **hai lần** còn phần
cuối epoch bị **bỏ qua**. Không lỗi nào báo, chỉ là dữ liệu bị học lệch.

Ba cơ chế phối hợp:

1. **Checkpoint lưu `buoc_trong_epoch`** — đã tiêu thụ bao nhiêu batch
2. **Thứ tự batch ghim theo cặp (seed, epoch)** — cùng seed và cùng số epoch thì
   thứ tự xáo trộn luôn giống hệt, không phụ thuộc máy nào chạy
3. **Tua nhanh khi phục hồi** — bỏ qua đúng số batch đã tiêu thụ rồi mới học tiếp

Cộng thêm trạng thái RNG được khôi phục nên mẫu dropout tiếp nối đúng mạch.

**Bằng chứng** — thí nghiệm giết phiên: lượt A chạy liền một mạch 60 bước; lượt B
cùng seed nhưng bị giết ở bước 20 và 40, mỗi lần dựng lại từ số 0 rồi nạp
checkpoint chạy tiếp.

| Chỉ số | Giá trị |
|---|---|
| Số bước so sánh | 60 |
| Chênh lệch trung bình | 0,0000% |
| **Chênh lệch lớn nhất** | **0,0000%** |
| Ngưỡng yêu cầu | 1,0% |

Đây là hình quan trọng nhất của cả đồ án, vì thiếu bất kỳ mảnh nào thì hai đường
**tách dần ra mà không có lỗi nào báo**.

### Tại sao chỉ cố định seed là chưa đủ?

Seed chỉ quyết định **điểm bắt đầu** của chuỗi số ngẫu nhiên. Nhưng bộ sinh số
ngẫu nhiên có **trạng thái tiến triển** — sau 10.000 lần gọi, nó đang ở một vị
trí rất xa điểm xuất phát.

Phục hồi mà chỉ đặt lại seed là **tua bộ sinh về đầu chuỗi**. Hệ quả: mẫu dropout
sau khi phục hồi lặp lại đúng những mẫu đã dùng ở đầu lượt chạy, không phải tiếp
nối. Mô hình thấy cùng một mẫu nhiễu hai lần.

Nên phải lưu **trạng thái RNG** chứ không chỉ seed.

Và còn bốn thứ nữa seed không kiểm soát nổi: trạng thái optimizer, scheduler,
GradScaler, và vị trí trong epoch. Xem câu tiếp theo.

Ngoài ra, tái lập được còn cần cố định seed cho **cả worker của DataLoader** —
mỗi worker là một tiến trình riêng với bộ sinh số riêng.

### Checkpoint lưu những gì? Chỉ lưu trọng số thì sao?

Bảy món, mỗi món thiếu đi hỏng một kiểu:

| Món | Vai trò | Thiếu thì hỏng thế nào |
|---|---|---|
| Trọng số mô hình | thứ đã học được | mất toàn bộ tiến trình |
| **Trạng thái optimizer** | AdamW giữ momentum và phương sai cho **từng** tham số | về 0 hết, vài bước đầu sau phục hồi bước sai cỡ, loss giật lên rõ rệt |
| Trạng thái scheduler | learning rate đang ở đâu trên lịch | lr nhảy về đầu lịch |
| Trạng thái GradScaler | hệ số giãn fp16, đang được chỉnh động | phải dò lại từ đầu, vài bước bị bỏ vì tràn |
| Số bước, số epoch | vị trí trong quá trình | không biết đang ở đâu, dừng sớm tính sai |
| Bản sao cấu hình | kiến trúc để dựng lại | dựng sai kiến trúc, nạp trọng số lỗi |
| Seed + trạng thái RNG | dropout, thứ tự batch | mẫu nhiễu và thứ tự dữ liệu khác đi |

**Món nặng nhất là trạng thái optimizer** — chiếm hơn nửa dung lượng file 549 MB,
vì AdamW lưu hai số thực cho mỗi tham số trong 47,96 triệu tham số.

**Điểm mấu chốt:** thiếu bất kỳ món nào cũng khiến phục hồi sai **mà không có lỗi
nào báo ra**. Huấn luyện vẫn chạy, loss vẫn trông hợp lý, chỉ là nó không còn là
cùng một lượt chạy nữa. Chỉ so hai đường loss mới phát hiện được.

Ngược lại, khi **chỉ đi dịch** thì 6 món kia hoàn toàn vô dụng — đó là lý do gói
gửi cho người làm frontend chỉ cần khoá `model`.

**Ghi file an toàn:** ghi ra file tạm rồi mới đổi tên. Ghi đè trực tiếp mà phiên
chết đúng lúc đang ghi thì mất **cả bản cũ lẫn bản mới**.

---

## Nhóm 3 — Quá khớp

### Train loss 0,9983 nhưng dev loss 2,3866 nói lên điều gì?

Chênh **1,39**. Con số tuyệt đối không quan trọng bằng **hình dạng hai đường**:

- loss train **tiếp tục giảm**
- loss dev **ngừng giảm**, đứng ở 2,3866

Giai đoạn đầu hai đường cùng đi xuống — mô hình đang học quy luật dịch, và quy
luật đó đúng cho cả dữ liệu chưa thấy. **Đúng lúc chúng tách ra là lúc mô hình
chuyển từ học quy luật sang học thuộc lòng tập train.** Thuộc bài thì điểm bài
cũ tăng mãi, còn điểm bài chưa từng thấy đứng yên.

Nguyên nhân đoán được từ trước: 131.339 cặp câu ít hơn WMT hàng trăm lần, mà mô
hình có 47,96 triệu tham số — thừa sức thuộc lòng.

### Bằng chứng nào để kết luận đã quá khớp?

Ba bằng chứng độc lập:

1. **Khoảng cách train–dev** 0,998 so với 2,387
2. **Hình dạng đường loss** — slide 16, đường train xuống đều tới ~1,0 trong khi
   đường dev tốt nhất (gạch đứt) nằm phẳng ở 2,3866
3. **Dừng sớm đã kích hoạt** — 5 lần đánh giá liên tiếp không cải thiện. Đây là
   bằng chứng mạnh nhất vì nó là cơ chế **tự động phát hiện**, không phải suy luận

### Biết dữ liệu nhỏ, sao vẫn train tới 17.000 bước / 98 epoch?

Đây là câu dễ bị hiểu thành "biết rồi mà kệ". Không phải.

Nhóm chuẩn bị **ba lớp phòng thủ từ trước khi bấm nút chạy**:

1. **Dropout 0,3** — cao gấp ba mức thông thường, đặt vậy vì biết trước dữ liệu ít
2. **Dừng sớm** sau 5 lần đánh giá không tiến
3. **`tot_nhat.pt` lưu bản TỐT NHẤT theo loss dev**, không phải bản cuối

Lớp thứ ba là then chốt: **quá khớp ở phần đuôi không làm hỏng sản phẩm giao
nộp**, vì bản được giữ là bản tại đáy đường dev.

Và về nguyên tắc: **không thể biết đáy ở đâu nếu chưa đi qua nó.** Dừng sớm hoạt
động đúng theo định nghĩa đó — chạy tiếp cho tới khi chắc chắn không còn cải
thiện, rồi quay lại lấy bản tốt nhất. Mấy nghìn bước "thừa" chính là cái giá để
**biết** mình đã tìm được đáy, thay vì đoán.

Nếu dừng ở epoch 40 vì "sợ quá khớp" thì không ai chứng minh được epoch 41 không
tốt hơn.

### Dừng sớm thực sự giải quyết quá khớp thế nào?

Nói cho chính xác: **dừng sớm không ngăn quá khớp xảy ra — nó ngăn quá khớp làm
hỏng sản phẩm.**

Cơ chế gồm hai nửa, và nửa thứ hai mới là phần quan trọng:

1. Theo dõi loss dev; 5 lần đánh giá liên tiếp không cải thiện thì dừng
2. **Bản giao nộp là bản tại đáy đường dev**, không phải bản cuối cùng

Nửa thứ nhất tiết kiệm GPU. Nửa thứ hai bảo vệ chất lượng. Kể cả nếu chạy tiếp
tới 50.000 bước, `tot_nhat.pt` vẫn giữ bản ở đáy.

Đây là dạng điều hoà **không đụng vào hàm mất mát** — khác hẳn dropout hay weight
decay, nó chỉ chọn thời điểm dừng.

### Sao không dùng label smoothing hoặc tăng dropout ngay từ đầu?

**Dropout thì có tăng ngay từ đầu** — 0,3 thay vì 0,1 tiêu chuẩn, chính vì biết
trước dữ liệu nhỏ.

**Label smoothing thì cố ý để 0,0.** Bài báo 2017 dùng 0,1, nhưng nhóm đặt nguyên
tắc: **mọi lựa chọn kiến trúc và công thức huấn luyện phải chứng minh bằng số
liệu của chính nhóm, không lấy mặc định của bài báo làm kết luận.**

Label smoothing được để dành cho thí nghiệm **A3** — bật so với tắt, đo BLEU,
rồi mới quyết định. Sau khi thầy rút phạm vi xuống còn ba thí nghiệm thì A3
không chạy, nên nó vẫn ở 0,0.

Trả lời trung thực: *"Chúng em để 0,0 làm mặc định để ablation quyết định, chứ
không phải vì nghĩ nó vô dụng. Thí nghiệm đó bị cắt do ngân sách GPU."*

### Dropout 0,3 dựa trên thí nghiệm hay chỉ là kinh nghiệm?

**Trả lời thẳng: chưa có thí nghiệm đo BLEU.**

Bảng khảo sát có đo ba mức dropout nhưng **chỉ đo thời gian**:

| Dropout | Giây/bước | VRAM |
|---|---|---|
| 0,1 | 0,3812 | 8.112 |
| **0,3** | **0,3710** | **8.112** |
| 0,5 | 0,3765 | 8.112 |

Thời gian gần như y hệt, nên bảng này chỉ chứng minh **chọn mức nào cũng không
tốn thêm**, chứ không chứng minh 0,3 là tốt nhất về chất lượng.

Lý do chọn 0,3 là **lập luận** chứ không phải **đo đạc**: 131k cặp câu là rất ít,
mô hình 48 triệu tham số dư sức thuộc lòng, nên cần điều hoà mạnh hơn mức tiêu
chuẩn.

Đây là một trong bốn chỗ yếu của nhóm. Nói thẳng ra sẽ ghi điểm hơn là lấp liếm.

### Nếu train loss giảm mà dev loss tăng thì chọn checkpoint nào?

**Checkpoint tại điểm dev loss thấp nhất** — và hệ thống đã tự làm điều đó.

`tot_nhat.pt` được ghi đè **mỗi khi** loss dev phá kỷ lục, nên tới cuối lượt chạy
nó luôn là bản tại đáy. Bản cuối cùng nằm ở `moi_nhat.pt`, chỉ dùng để chạy tiếp.

Lý do không bao giờ chọn theo train loss: train loss đo mức **thuộc bài**, còn
điều quan tâm là khả năng dịch câu **chưa từng thấy**. Chọn theo train loss là
chọn bản thuộc bài nhất, tức bản tệ nhất.

Đây chính xác là chuyện đã xảy ra: lượt chính có train loss 0,998 ở bước 17.000
nhưng bản được giữ là bản có dev loss 2,3866 ở một bước sớm hơn.

---

## Nhóm 4 — Transformer

### Tại sao chọn RoPE thay vì sinusoidal?

Nói cho chính xác: **nhóm chọn RoPE làm mặc định, và chuẩn bị thí nghiệm A4 để
kiểm chứng — nhưng A4 bị cắt khỏi phạm vi.**

Lý do chọn ban đầu, về mặt cơ chế:

**Sinusoidal cộng vector vị trí vào embedding.** Thông tin vị trí trộn vào nội
dung ngay từ đầu, rồi đi qua 6 lớp biến đổi — tới lớp cuối nó đã bị pha loãng.
Và nó mã hoá **vị trí tuyệt đối**: token ở vị trí 5 luôn nhận cùng một vector,
bất kể câu dài ngắn.

**RoPE xoay vector q và k theo vị trí.** Điểm tinh tế: sau khi xoay, tích vô
hướng giữa q ở vị trí *m* và k ở vị trí *n* **chỉ còn phụ thuộc hiệu *m − n***.
Tức RoPE mã hoá **khoảng cách tương đối** một cách tự nhiên, không cần tham số
học thêm.

Khoảng cách tương đối hợp với dịch máy hơn: quan hệ "tính từ đứng ngay sau danh
từ" là quan hệ **tương đối**, đúng dù cụm đó nằm đầu hay cuối câu.

RoPE cũng được áp lại **ở mỗi lớp**, nên thông tin vị trí không bị pha loãng dần.

**Thừa nhận giới hạn:** nhóm **không có số** chứng minh RoPE hơn sin-cos trên bài
toán này. File cấu hình A4 đã viết sẵn, đổi một dòng YAML là chạy được.

### RoPE thực sự mã hoá vị trí như thế nào?

Chia vector thành từng **cặp chiều**, coi mỗi cặp là một điểm trên mặt phẳng, rồi
**xoay** điểm đó một góc tỉ lệ với vị trí trong câu.

Mỗi cặp chiều xoay với **tần số khác nhau**: cặp đầu xoay nhanh (bắt quan hệ gần),
cặp sau xoay chậm (bắt quan hệ xa). Giống kim giây và kim giờ cùng chỉ một thời
điểm ở hai độ phân giải.

Tính chất then chốt: xoay q một góc *mθ* và k một góc *nθ* thì tích vô hướng của
chúng chỉ phụ thuộc *(m − n)θ*. **Vị trí tuyệt đối triệt tiêu, chỉ còn khoảng
cách.**

Hai điểm cài đặt quan trọng:

- **Bảng góc quay phải tính ở float32 rồi mới ép kiểu.** Tính thẳng ở fp16 là
  mất độ phân giải góc và **RoPE hỏng âm thầm** — mô hình vẫn chạy, vẫn ra câu,
  chỉ là kém đi mà không có gì báo
- **Không gọi `model.half()`** vì nó ép luôn bảng góc quay xuống fp16. Phải dùng
  `torch.autocast`

### Tại sao RoPE dùng ở self-attention nhưng không ở cross-attention?

Vì ở cross-attention, **"khoảng cách" giữa query và key là một khái niệm vô
nghĩa**.

Self-attention: query và key **cùng một câu**. Hỏi "từ thứ 5 cách từ thứ 3 bao
xa" là câu hỏi có nghĩa — đáp án là 2 vị trí.

Cross-attention: query nằm ở **câu tiếng Việt**, key nằm ở **câu tiếng Anh**. Hỏi
"vị trí 5 của câu Việt cách vị trí 3 của câu Anh bao xa" là câu hỏi **không có
nghĩa gì cả**. Hai chuỗi khác nhau, thứ tự từ khác nhau (tiếng Việt tính từ đứng
sau danh từ, tiếng Anh đứng trước), độ dài khác nhau.

Áp RoPE vào đây là **bơm nhiễu có cấu trúc** vào mô hình. Và nó **vẫn chạy** —
không lỗi, không NaN, chỉ là dịch kém hơn. Đúng loại lỗi khó phát hiện nhất.

Nhóm có hẳn **bài kiểm tra số 11** canh riêng chuyện này.

### Tại sao dùng RMSNorm thay vì LayerNorm?

**Vì nó nhanh hơn, chứ không phải vì nó dịch tốt hơn.** Nhóm có số của chính mình.

| | BLEU tst2013 | giây/bước |
|---|---|---|
| RMSNorm | 28,56 ± 0,18 | **1,511** |
| LayerNorm | 28,60 ± 0,27 | 1,598 |

Cơ chế: LayerNorm làm hai việc — **trừ trung bình** rồi **chia độ lệch chuẩn**.
RMSNorm **bỏ hẳn bước trừ trung bình**, chỉ chia cho căn trung bình bình phương.
Ít phép tính hơn nên nhanh hơn.

Zhang & Sennrich (2019) lập luận rằng cái làm LayerNorm hiệu quả là bước **chia
tỉ lệ**, không phải bước **dời tâm** — và số đo của nhóm khớp với lập luận đó:
bỏ bước dời tâm mà chất lượng không đổi.

### Chỉ khác 0,04 BLEU sao lại kết luận RMSNorm tốt hơn?

**Nhóm KHÔNG kết luận RMSNorm cho chất lượng tốt hơn.** Nếu slide hay lời nói tạo
ra ấn tượng đó thì phải sửa ngay.

Kết luận đúng gồm **hai vế tách bạch**:

**Vế 1 — chất lượng: không phân biệt được.**
Chênh lệch 0,04 BLEU, trong khi dao động giữa hai seed của **chính từng cấu hình**
là 0,18 và 0,27. Tín hiệu chìm dưới nhiễu. Khi tín hiệu nhỏ hơn nhiễu, kết luận
"không phân biệt được" là kết luận **hợp lệ**, không phải thí nghiệm thất bại.

**Vế 2 — tốc độ: khác biệt thật.**
5,7% nhanh hơn, trong khi dao động thời gian giữa hai seed cùng cấu hình chỉ
**0,4% và 0,2%** (4541,3 so với 4525,2; 4787,9 so với 4799,2). Khoảng cách lớn
hơn nhiễu hơn chục lần.

**Nên đây là đánh đổi chất lượng đổi tốc độ, mà phía chất lượng hoà nhau.** Chọn
bản nhanh hơn là lựa chọn hiển nhiên khi hai bên dịch ngang nhau.

Cách nói chuẩn khi bảo vệ:

> "Chúng em không nói RMSNorm dịch tốt hơn. Chúng em đo được chất lượng không
> phân biệt được, và RMSNorm nhanh hơn 5,7%. Khi chất lượng hoà thì chọn bản
> nhanh hơn."

### Tại sao SwiGLU cần ba ma trận W₁, W₂, W₃?

Vì nó có **cơ chế cổng**, và cổng cần một nhánh riêng.

FFN thường có hai bước: nở rộng (W₁) → phi tuyến → thu hẹp (W₂).

SwiGLU tách nhánh nở rộng làm **hai đường song song**:

```
SwiGLU(x) = ( Swish(x·W₁) ⊙ x·W₃ ) · W₂
              └─ đường cổng ─┘  └ nội dung ┘
```

- `x·W₁` qua hàm Swish tạo ra **cổng** — quyết định thông tin nào được đi qua
- `x·W₃` mang **nội dung**
- Nhân từng phần tử (⊙): cổng gần 0 thì chặn, gần 1 thì cho qua
- `W₂` thu hẹp về lại d_model

Điểm hay: cổng **phụ thuộc đầu vào**. Cùng một vị trí, tuỳ ngữ cảnh mà nó cho qua
thông tin khác nhau. ReLU thì chỉ có một quy tắc cố định — âm thì cắt, dương thì
giữ.

Shazeer (2020) đo trên nhiều tác vụ và thấy các biến thể GLU đều hơn ReLU ở cùng
số tham số.

**Thừa nhận giới hạn:** đây là thí nghiệm **A5**, cũng bị cắt. Nhóm chưa có số
chứng minh SwiGLU hơn ReLU trên bài toán này.

### Tại sao d_ff = 688 thay vì 2048?

**Đây là con số dễ bị hỏi nhất trong cả bảng, và có lý do rất cụ thể.**

Vì SwiGLU dùng **ba** ma trận còn ReLU dùng **hai**, nên để d_ff bằng nhau thì
SwiGLU tự động nhiều tham số hơn 50%. Muốn so sánh công bằng phải **cân số tham
số**, không phải cân d_ff:

```
SwiGLU:  3 × 512 × 688  = 1.056.768
ReLU:    2 × 512 × 1024 = 1.048.576     chênh 0,8%
```

**688 được chọn để SwiGLU có đúng bằng số tham số của một FFN ReLU 1024 tiêu
chuẩn.** Không làm vậy thì thí nghiệm A5 đo **kích thước** chứ không đo **kiến
trúc** — và mất sạch ý nghĩa.

Còn vì sao 1024 chứ không phải 2048: bảng khảo sát cho thấy d_ff 1024 tốn
54,15 triệu tham số ở 0,3951 giây/bước, so với 688 là 47,96 triệu ở 0,3710.
Chọn mức nhỏ hơn để vừa ngân sách 25 giờ GPU. Với ngữ liệu 131k câu thì mô hình
lớn hơn cũng chỉ quá khớp nhanh hơn.

### Tại sao dùng Pre-Norm thay vì Post-Norm?

**Pre-Norm:** `x + SubLayer(Norm(x))` — chuẩn hoá **trước** khối con
**Post-Norm:** `Norm(x + SubLayer(x))` — chuẩn hoá **sau**, như bài báo 2017

Khác biệt then chốt nằm ở **đường residual**. Ở Pre-Norm, đường `x + ...` chạy
thẳng từ đầu tới cuối mà **không bị lớp chuẩn hoá nào chắn ngang**. Gradient đi
ngược về các lớp đầu không bị suy giảm.

Ở Post-Norm, mỗi lớp có một lần chuẩn hoá chắn trên đường residual. Qua 6 lớp,
gradient bị bóp lại 6 lần — nên cần **warmup dài** để giai đoạn đầu không nổ.

Xiong và cộng sự (2020) chứng minh chính xác điều này: **Post-Norm phụ thuộc nặng
vào warmup, Pre-Norm thì không.**

**Và nhóm có bằng chứng thực nghiệm bất ngờ ủng hộ điều đó:** thí nghiệm A0 dùng
Post-Norm **không huấn luyện được** ở ngân sách 3.000 bước, sau ba lần thử — vì
riêng warmup của công thức 2017 đã dài hơn cả ngân sách. Xem nhóm câu hỏi 6.

Đây có lẽ là kết quả đáng giá nhất của cả phần ablation: nhóm không định chứng
minh điều này, nhưng dữ liệu tự nói ra.

**Hệ quả cài đặt:** Pre-Norm **bắt buộc phải có một lớp chuẩn hoá cuối** trước
lớp xuất. Vì đường residual không bao giờ được chuẩn hoá nên giá trị phình dần
qua 6 lớp. Quên lớp này không báo lỗi, chỉ làm mô hình mất ổn định. Bài kiểm tra
số 10 canh riêng chuyện này.

### Chia sẻ trọng số embedding và lớp xuất có làm giảm khả năng biểu diễn không?

**Về lý thuyết là có giảm** — ba ma trận độc lập bao giờ cũng biểu diễn được
nhiều hơn một ma trận dùng chung. Nhưng trong bối cảnh này, **đổi lại là lợi**.

Ba lý do:

**Tiết kiệm 16,4 triệu tham số.** Bảng đếm cho thấy lớp xuất tốn **0 tham số**.
Với 131k cặp câu, ít tham số hơn nghĩa là ít quá khớp hơn — mà quá khớp đúng là
bài toán trung tâm của đồ án.

**Có cơ sở ngôn ngữ học.** Ma trận embedding ánh xạ token → vector; ma trận xuất
ánh xạ vector → điểm số cho từng token. Hai phép này là **nghịch đảo của nhau**,
nên dùng chung một biểu diễn là hợp lý.

**Từ vựng dùng chung En + Vi nên càng hợp.** Tên riêng, con số, thuật ngữ kỹ thuật
xuất hiện y hệt ở cả hai ngôn ngữ — chia sẻ giúp chúng có cùng một biểu diễn thay
vì học hai lần.

Với ngữ liệu nhỏ, **giảm tham số là ưu tiên cao hơn tăng sức biểu diễn**. Nếu có
hàng triệu cặp câu thì cân nhắc lại sẽ hợp lý.

---

## Nhóm 5 — Kết quả BLEU

### BLEU 29,53 có thực sự chứng minh mô hình dịch tốt không?

**Không, và phải nói rõ điều đó.**

Theo thang của chính đề tài (≥19 chấp nhận, ≥22 tốt, ≥29 rất tốt) thì nó vượt mốc
rất tốt. Nhưng BLEU có ba giới hạn:

**BLEU đo trùng khớp n-gram với MỘT bản tham chiếu duy nhất.** Một bản dịch hoàn
toàn đúng nhưng dùng từ khác sẽ bị chấm thấp oan. Tiếng Việt có rất nhiều cách
diễn đạt tương đương nên điều này càng đáng kể.

**BLEU không đo nghĩa, không đo ngữ pháp.** Nó đếm chuỗi từ trùng nhau.

**BLEU không so được giữa các cách cài đặt khác nhau** — đó chính là lý do tồn
tại của sacrebleu và chuỗi chữ ký.

**Và có bằng chứng thực tế:** khi thử nghiệm với người dùng, mô hình dịch
"badminton" thành "những thứ vớ vẩn", dịch "buffet" thành "thịt bò", và **lặp câu
khi câu nhập thiếu dấu chấm cuối**. BLEU 29,53 tồn tại song song với những lỗi đó.

Cách phát biểu trung thực:

> "29,53 là con số vững cho bộ dữ liệu và ngân sách này. Nó có nghĩa mô hình dịch
> tốt **văn phong TED** — không có nghĩa nó là một hệ dịch mạnh nói chung."

*(Lỗi lặp khi thiếu dấu chấm có nguyên nhân cụ thể: phía câu nguồn không có token
`<eos>`, nên dấu chấm cuối đang làm nhiệm vụ đánh dấu hết câu. Bỏ nó đi thì
decoder không biết khi nào phát `<eos>`. Vá được ở tầng giao diện bằng cách tự
thêm dấu chấm nếu thiếu.)*

### Tại sao chỉ dùng tst2013 để báo cáo kết quả cuối?

Vì **tst2013 là tập test chuẩn của cộng đồng cho cặp Anh–Việt IWSLT 2015**. Dùng
đúng tập đó thì người khác so được với kết quả của nhóm.

tst2012 làm tập dev — dùng để chọn checkpoint tốt nhất và kích hoạt dừng sớm.

**Lưu ý quan trọng:** slide đang ghi tst2013 *"chỉ dùng một lần"* — câu đó **không
đúng**, xem mục [Bốn chỗ nhóm yếu](#bốn-chỗ-nhóm-yếu-phải-biết-trước). Phải sửa
trước khi bảo vệ.

### Tại sao không dùng tập test trong lúc huấn luyện / tinh chỉnh?

Vì làm vậy là **biến tập test thành tập dev thứ hai**, và điểm báo cáo bị thổi
phồng.

Cơ chế: nếu thử 20 cấu hình rồi chọn cấu hình có điểm test cao nhất, thì đã
**chọn theo tập test**. Cấu hình đó có điểm cao một phần vì nó thật sự tốt, một
phần vì nó **may mắn hợp với đúng 1.268 câu đó**. Điểm báo cáo không còn phản ánh
khả năng dịch câu chưa từng thấy.

Đây là dạng quá khớp ở **cấp quy trình**, không phải cấp mô hình, và nguy hiểm
hơn vì không có đường loss nào để nhìn ra.

Nhóm chọn mọi thứ dựa trên **tst2012**: checkpoint tốt nhất theo loss dev, dừng
sớm theo loss dev.

### Nếu đã dùng tst2013 để quyết định kiến trúc thì còn gọi là test set được không?

**Câu hỏi rất đúng chỗ, và phải trả lời trung thực.**

Sự thật: tst2013 đã được **chấm nhiều lần** — 6 lượt ablation, cộng lần so
greedy/beam. Nếu nhóm đã nhìn điểm test rồi đổi kiến trúc theo thì đúng là nó
không còn là test set đúng nghĩa.

**Nhưng điều nhóm làm không phải vậy:**

- Chọn checkpoint: theo **loss dev** (tst2012), tự động, không ai nhìn điểm test
- Dừng sớm: theo **loss dev**
- Chọn siêu tham số: theo bảng khảo sát tốc độ và VRAM, trước khi có điểm test nào
- Kết luận A1: tính trên **cả dev lẫn test**, và kết luận giống nhau ở cả hai
  (chênh lệch chìm dưới nhiễu ở cả hai tập)

Điểm test được tính **sau khi** mọi quyết định đã chốt, và dùng để **báo cáo**.

Cách trả lời:

> "Chúng em có chấm tst2013 cho từng lượt ablation, nên nói 'chỉ dùng một lần' là
> không chính xác — chỗ đó trên slide chúng em ghi sai. Nhưng tst2013 không được
> dùng để **chọn** bất cứ thứ gì: checkpoint và dừng sớm đều theo tst2012, siêu
> tham số chốt trước khi có điểm test. Nên điểm báo cáo vẫn giữ được ý nghĩa."

Thừa nhận sai sót nhỏ rồi giải thích rõ ranh giới sẽ đáng tin hơn nhiều so với
cãi rằng mình không sai.

### Beam = 4 tăng BLEU 28,74 → 29,53 nhưng thời gian tăng 1,8 lần?

**Vì sao chậm hơn:** mỗi bước sinh từ, beam phải chạy decoder cho **4 giả thuyết**
thay vì 1.

**Vì sao chỉ 1,8 lần chứ không phải 4 lần:**

- 4 nhánh được xử lý như **một batch**, GPU tính song song nên phần lớn chi phí
  bị hấp thụ
- **KV cache** loại bỏ phần tính lại dư thừa cho cả 4 nhánh

Số đo đầy đủ:

| Tập | Cách | BLEU | chrF++ | Giây |
|---|---|---|---|---|
| tst2012 (dev) | Greedy | 25,41 | 44,77 | 8,9 |
| tst2012 (dev) | Beam = 4 | 26,59 | 45,75 | 15,7 |
| tst2013 (test) | Greedy | 28,74 | 47,62 | 9,8 |
| tst2013 (test) | **Beam = 4** | **29,53** | **48,41** | 17,2 |

Đánh đổi thực tế: **+0,80 BLEU đổi lấy 1,8 lần thời gian** trên test, **+1,18** trên
dev. Giao diện cần phản hồi tức thì thì dùng greedy; cần chất lượng thì dùng beam.

### Tại sao Beam Search cho kết quả tốt hơn Greedy?

**Greedy** mỗi bước chọn từ có xác suất cao nhất **ngay tại bước đó**, rồi đi
tiếp, không bao giờ quay lại. Vấn đề: từ tốt nhất ở bước 1 có thể dẫn tới một
**câu** có xác suất tổng thấp. Chọn sai một bước là hỏng cả câu, không sửa được.

**Beam = 4** giữ song song 4 phương án tốt nhất, mỗi bước mở rộng cả 4 rồi giữ
lại 4 tốt nhất trong các nhánh mới. Cuối cùng chọn **chuỗi có tổng điểm cao
nhất**. Nhờ vậy cứu được tình huống bước đầu hơi kém nhưng về sau tốt hơn nhiều.

**Hệ số phạt độ dài 1,0:** không có nó thì beam thiên vị câu ngắn — mỗi token
thêm vào nhân thêm một xác suất nhỏ hơn 1, nên câu càng ngắn tổng điểm càng cao.
Hệ số phạt chuẩn hoá điểm theo độ dài.

**Beam lớn hơn có tốt hơn không?** Không tuyến tính. Beam quá lớn thường **giảm**
chất lượng vì nó tìm ra những chuỗi xác suất cao nhưng ngắn và nhạt. Nhóm dùng 4
theo yêu cầu TASK 19 và không tinh chỉnh con số này để vượt ngưỡng — làm vậy là
bịa kết quả.

### chrF++ bổ sung thông tin gì mà BLEU không có?

**BLEU đếm theo từ. chrF++ đếm theo ký tự** (cộng thêm một phần n-gram từ).

Khác biệt quan trọng với tiếng Việt:

**chrF++ cho điểm từng phần khi dịch gần đúng.** Dịch "người nghiên cứu" trong khi
đáp án là "nhà nghiên cứu" — BLEU coi là sai hoàn toàn ở mọi n-gram chứa từ đó,
chrF++ vẫn ghi nhận phần lớn ký tự trùng khớp.

**chrF++ nhạy với hình thái từ.** Sai một dấu thanh, sai một tiền tố — BLEU mất
trọn n-gram, chrF++ chỉ mất vài ký tự.

**chrF++ ít phụ thuộc cách tách từ.** Tiếng Việt không có ranh giới từ rõ ràng
("nhà nghiên cứu" là một từ hay ba từ?), nên thước đo theo ký tự ổn định hơn.

Dùng cả hai để tránh trường hợp một thước đo cho kết luận sai. Trong kết quả của
nhóm, hai thước đo **nhất quán**: beam hơn greedy ở cả BLEU (+0,80) lẫn chrF++
(+0,78). Nếu chúng mâu thuẫn thì mới phải điều tra.

### BLEU 29,53 gọi là "rất tốt" theo tiêu chuẩn nào?

Theo **thang do chính nhóm đặt ra từ đầu đồ án**, ghi ở slide mục tiêu:

| Mức | BLEU tst2013 |
|---|---|
| Tối thiểu chấp nhận | ≥ 19 |
| Tốt | ≥ 22 |
| **Rất tốt** | **≥ 29** |

Thang này đặt **trước khi** huấn luyện, dựa trên các kết quả đã công bố cho cặp
Anh–Việt IWSLT. Đặt trước rồi mới đo là điểm quan trọng — đặt sau khi có kết quả
thì thang nào cũng "đạt".

**Nhưng phải nói rõ đây là thang nội bộ**, không phải chuẩn quốc tế. Không có
ngưỡng BLEU phổ quát nào cho "tốt" — nó phụ thuộc cặp ngôn ngữ, miền dữ liệu, và
cách chấm.

Mốc đối chiếu thật từ chiến dịch IWSLT 2015:

| Hệ | Anh→Việt |
|---|---|
| Moses (Tran Hong Viet và cs.) | 23,15 |
| Hiero (Trieu Hai Long và cs.) | 21,48 |
| baseline của workshop | 27,01 |
| Phrase-based out-of-domain (Trieu) | 28,17 |
| **Của nhóm** | **29,53** |

**Không được kết luận "tốt hơn"** vì ba lý do: khác tập test (họ tst2015, nhóm
tst2013), khác cách tính BLEU (họ tách từ bằng VnTokenizer trước khi chấm), khác
điều kiện dữ liệu (họ dùng thêm dữ liệu ngoài, nhóm chỉ dùng trong ràng buộc).

Cách nói đúng: *"ngang tầm hệ tốt nhất của chiến dịch IWSLT 2015, trong khi chỉ
dùng dữ liệu trong ràng buộc."* Và nói thêm: cả hai bài đó đều là **SMT dựa trên
cụm từ**, thế hệ trước nơ-ron — Transformer thắng SMT là chuyện đã biết từ 2017.

---

## Nhóm 6 — Bắt lỗi phương pháp

### 3.000 bước có đủ để kết luận RMSNorm tốt hơn LayerNorm không?

**Trước hết phải đính chính tiền đề: nhóm KHÔNG kết luận RMSNorm tốt hơn.** Kết
luận là chất lượng **không phân biệt được**, và RMSNorm nhanh hơn 5,7%.

Với kết luận đó thì 3.000 bước **đủ**, vì hai lý do:

**Kết luận "không phân biệt được" chỉ cần so tín hiệu với nhiễu.** Chênh lệch
0,04 nhỏ hơn nhiễu 0,18 và 0,27. Điều này đúng bất kể ngân sách bao nhiêu bước.

**Kết luận về tốc độ không phụ thuộc số bước.** Giây/bước là đại lượng cục bộ,
đo ở 3.000 bước hay 30.000 bước cũng vậy.

**Nhưng phải nêu giới hạn:** kết luận này gắn với **ngân sách 3.000 bước, dữ liệu
này, kích thước mô hình này**. Ở ngân sách dài hơn hoàn toàn có thể khác. Chính
A0 là ví dụ sống: nó hỏng **chỉ vì** ngân sách ngắn.

### Tại sao mỗi ablation chỉ 2 seed? Có đủ thống kê không?

**Trả lời thẳng: không đủ để phát biểu thống kê chặt chẽ.**

Hai seed cho biết **khoảng dao động**, nhưng không đủ để tính độ lệch chuẩn đáng
tin, khoảng tin cậy, hay giá trị p. Muốn kiểm định thống kê đàng hoàng cần ít
nhất 5 seed.

**Nhưng hai seed vẫn đủ cho kết luận nhóm đưa ra**, vì kết luận đó không đòi hỏi
kiểm định: nhóm chỉ cần chứng minh **tín hiệu nhỏ hơn nhiễu**. Chênh lệch 0,04 so
với dao động 0,18 và 0,27 — kết luận "không phân biệt được" đứng vững mà không
cần giá trị p.

Nếu chênh lệch là 0,5 BLEU và nhóm muốn khẳng định "thật sự khác" thì lúc đó 2
seed **không đủ**, phải chạy thêm.

Lý do dừng ở 2: ngân sách GPU. 14 lượt × 3.000 bước đã là 17 giờ; 5 seed sẽ thành
hơn 40 giờ, vượt quota 30 giờ/tuần.

### Dao động ±0,18 mà chênh lệch chỉ 0,04, sao vẫn đưa ra kết luận?

Vì **đó chính là kết luận**: khi tín hiệu nhỏ hơn nhiễu, câu trả lời đúng là
*"không phân biệt được"*, và đó là một kết quả hợp lệ.

Cái sai là làm ngược lại — thấy 0,04 rồi tuyên bố "RMSNorm hơn". Nhóm **không**
làm vậy.

Giá trị của thí nghiệm nằm ở chỗ nó **ngăn một kết luận sai**. Nếu không chạy A1,
nhóm sẽ giữ RMSNorm với lý do "bài báo nói nó tốt hơn" — một lý do không kiểm
chứng. Sau A1, nhóm biết lý do thật là **tốc độ**, và có số chứng minh.

Nói cách khác: thí nghiệm cho kết quả "hoà" vẫn là thí nghiệm thành công, miễn là
báo cáo trung thực rằng nó hoà.

### A0 lỗi 3 lần — lỗi ở cài đặt hay ở thiết kế thí nghiệm?

**Ở thiết kế thí nghiệm. Cài đặt không sai, và có bằng chứng.**

Cổng chặn học thuộc 50 câu chạy với **đúng cấu hình vanilla**:

| bước | 50 | 150 | 250 | 350 | 450 | 500 |
|---|---|---|---|---|---|---|
| loss | 6,49 | 4,13 | 1,93 | 0,31 | 0,090 | 0,062 |

**BLEU trên chính 50 câu đó: 100,00.** Kiến trúc vanilla học thuộc hoàn hảo. Một
kiến trúc cài sai thì không thể làm được điều này.

Ba lần hỏng đều do **lịch learning rate**, mỗi lần một lý do khác:

| Lượt | warmup | lr đỉnh | Hỏng vì |
|---|---|---|---|
| 1 | 4.000 | 6,99e-4 | warmup dài hơn cả ngân sách 3.000 bước, lr chưa bao giờ lên tới đỉnh |
| 2 | 120 | **4,03e-3** | rút ngắn warmup vô tình đẩy đỉnh lên 5,8 lần |
| 3 | 120 + ghim đỉnh 7e-4 | 7,0e-4 | đỉnh đúng, nhưng dốc quá gắt cho Post-Norm |

**Gốc rễ:** lịch Noam **buộc chặt** warmup với learning rate đỉnh:

```
lr_đỉnh = d_model^(-0.5) × warmup^(-0.5)
```

Nên không thể rút ngắn warmup mà giữ nguyên đỉnh. Con số 4.000 của bài báo cho ra
đỉnh **6,99e-4** — gần như trùng khít learning rate 7e-4 mà nhóm dùng. Không phải
trùng hợp.

**Và kết luận cuối cùng tự nó là một kết quả đáng báo cáo:** Post-Norm cần warmup
dài hơn cả ngân sách 3.000 bước. Đó đúng là điều Xiong và cộng sự (2020) chỉ ra.
**Việc không huấn luyện được công thức 2017 ở ngân sách này chính là bằng chứng
thực nghiệm ủng hộ lựa chọn Pre-Norm của nhóm.**

### Baseline vanilla không chạy được — nhóm có thiếu baseline quan trọng không?

**Có, và phải thừa nhận.** Không có số của A0 thì không trả lời được định lượng
câu "toàn bộ gói cải tiến đóng góp bao nhiêu BLEU".

Nhưng ba điểm cần nói rõ:

**Baseline để so sánh ablation là "đối chứng", không phải A0.** Mỗi thí nghiệm
A1–A6 đổi **đúng một yếu tố** so với đối chứng. Cấu trúc so sánh đó vẫn nguyên
vẹn. A0 trả lời một câu hỏi **khác**: đổi cả gói cùng lúc thì sao.

**Việc A0 không chạy được tự nó là dữ liệu.** Nó cho biết công thức 2017 không
huấn luyện được ở ngân sách ngắn — một phát hiện có ích, khớp với lý thuyết đã
công bố.

**Có thể chạy được nếu tăng ngân sách.** Ở 100.000 bước như bài báo gốc thì warmup
4.000 chỉ chiếm 4% và Post-Norm hoạt động bình thường. Giới hạn là ngân sách GPU
chứ không phải bế tắc kỹ thuật.

Trả lời trung thực:

> "Đúng là chúng em thiếu con số của A0. Nhưng chúng em biết **vì sao** thiếu, đã
> chứng minh kiến trúc không sai bằng cổng chặn BLEU 100, và bản thân việc nó
> không chạy được ở ngân sách này đã ủng hộ lựa chọn Pre-Norm."

### "8 cấu hình đã khảo sát" là những cấu hình nào? Tiêu chí chọn?

Khảo sát trên Tesla T4, fp16, 30 bước mỗi cấu hình:

| Lớp | Head | d_ff | Dropout | Tham số | Giây/bước | VRAM (MB) |
|---|---|---|---|---|---|---|
| 4 | 4 | 688 | 0,3 | 37.432.320 | 0,2728 | 6.959 |
| 4 | 8 | 688 | 0,3 | 37.432.320 | 0,2839 | 7.025 |
| 6 | 4 | 688 | 0,3 | 47.955.968 | 0,3572 | 8.016 |
| **6** | **8** | **688** | **0,3** | **47.955.968** | **0,3710** | **8.112** |
| 6 | 8 | 512 | 0,3 | 44.711.936 | 0,3655 | 7.924 |
| 6 | 8 | 1024 | 0,3 | 54.149.120 | 0,3951 | 8.435 |
| 6 | 8 | 688 | 0,1 | 47.955.968 | 0,3812 | 8.112 |
| 6 | 8 | 688 | 0,5 | 47.955.968 | 0,3765 | 8.112 |

Đây là quét **một chiều quanh một cấu hình trung tâm**: đổi số lớp, đổi số head,
đổi d_ff, đổi dropout — mỗi lần một thứ.

**Tiêu chí chọn:**

1. **Nằm trong VRAM T4** — mọi cấu hình đều dưới 16 GB nên không loại được ai;
   nhưng mức 8.192 token/batch thì **tràn bộ nhớ**, loại ngay
2. **Nằm trong ngân sách 30 giờ GPU/tuần** — cấu hình chọn cho tổng ~25 giờ
3. **Sức biểu diễn tối đa trong hai ràng buộc trên** — chọn 6 lớp thay vì 4, 8
   head thay vì 4

**Phải nói rõ một giới hạn:** bảng này **chỉ đo tốc độ và bộ nhớ, không đo BLEU**.
Nên nó trả lời "cấu hình nào chạy được trong ngân sách", không trả lời "cấu hình
nào dịch tốt nhất". Đó là lý do slide gốc có câu *"nhanh nhất không đồng nghĩa
với tốt nhất"*.

*(Slide này đã bị cắt khỏi bài trình bày để vừa 15 phút, nhưng số liệu vẫn nằm
trong `results/khao_sat_cau_hinh.csv` nếu hội đồng hỏi tới.)*

### Thay đổi nhiều siêu tham số, làm sao đảm bảo không quá khớp vào tập test?

Đây là câu hỏi sắc nhất trong cả nhóm, vì nó nhắm vào quá khớp ở **cấp quy trình**
— thứ không có đường loss nào để nhìn ra.

Ba lớp bảo vệ nhóm đã dùng:

**Bảng khảo sát 8 cấu hình chốt TRƯỚC khi có bất kỳ điểm test nào.** Nó chỉ dùng
giây/bước và VRAM — hai đại lượng **không liên quan gì tới tập test**. Nên không
có đường nào để thông tin từ test rò vào lựa chọn kiến trúc.

**Mọi lựa chọn trong lúc huấn luyện đều theo tst2012.** Checkpoint tốt nhất chọn
theo loss dev, dừng sớm theo loss dev. Tự động, không ai nhìn điểm test để can
thiệp.

**Ablation cố định ngân sách bước trước, không tinh chỉnh theo kết quả.** Cả 6
lượt chạy đúng 3.000 bước. Nhóm không thử 2.000 rồi 4.000 rồi chọn con số cho
điểm đẹp nhất.

**Thừa nhận phần yếu:** tst2013 đã được chấm nhiều lần, nên về nguyên tắc nhóm
**đã thấy** điểm test nhiều lần trong quá trình làm. Dù không dùng nó để chọn gì,
việc nhìn thấy nhiều lần vẫn tạo ra một kênh rò rỉ mềm — người làm có thể vô thức
thiên lệch.

Cách phòng đúng chuẩn là khoá tập test lại và chỉ mở đúng một lần cuối. Nhóm
không làm được điều đó vì quy trình ablation tự động chấm cả hai tập cho mỗi lượt.

Trả lời trung thực:

> "Chúng em bảo vệ được ở mức: không quyết định nào dựa trên điểm test. Nhưng
> chúng em **không** bảo vệ được ở mức nghiêm ngặt nhất là chỉ mở tập test đúng
> một lần — quy trình ablation của chúng em chấm cả hai tập cho mỗi lượt. Nếu làm
> lại, chúng em sẽ tách riêng bước chấm test ra khỏi vòng lặp ablation."

Câu cuối quan trọng: nêu được **cách làm đúng hơn nếu làm lại** cho thấy nhóm hiểu
vấn đề chứ không chỉ biện hộ.

---

## Bảng số liệu tra nhanh

### Chấm điểm chính thức

Checkpoint `abl3000_base_seed42/tot_nhat.pt`, bước 3.000, loss dev 2,2767.

| Tập | Sinh câu | Số câu | BLEU | chrF++ | Giây |
|---|---|---|---|---|---|
| tst2012 (dev) | Greedy | 1.553 | 25,41 | 44,77 | 8,9 |
| tst2012 (dev) | Beam = 4 | 1.553 | 26,59 | 45,75 | 15,7 |
| tst2013 (test) | Greedy | 1.268 | 28,74 | 47,62 | 9,8 |
| **tst2013 (test)** | **Beam = 4** | 1.268 | **29,53** | **48,41** | 17,2 |

```
Chữ ký BLEU    nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
Chữ ký chrF++  nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

### Ablation

| Cấu hình | seed 42 | seed 1337 | trung bình | giây/bước |
|---|---|---|---|---|
| Đối chứng (RMSNorm) | 28,74 | 28,37 | 28,56 ± 0,18 | **1,511** |
| A1 (LayerNorm) | 28,87 | 28,32 | 28,60 ± 0,27 | 1,598 |

### Lượt huấn luyện chính

| Chỉ số | Giá trị |
|---|---|
| Tổng số bước | 17.000 |
| Số epoch | 98 |
| Loss train cuối | 0,9983 |
| Loss dev tốt nhất | 2,3866 (ppl 10,88) |
| Dừng sớm | có |
| Tốc độ | ~13.800 token/giây |

### Mô hình

| | |
|---|---|
| Tổng tham số | 47.955.968 |
| Embedding (dùng chung) | 16.384.000 |
| Encoder 6 lớp | 12.638.720 |
| Decoder 6 lớp | 18.933.248 |
| Lớp xuất | 0 (chia sẻ trọng số) |

### Dữ liệu

| Split | Trước | Sau | Bị loại |
|---|---|---|---|
| train | 133.317 | 131.339 | 1.978 (1,48%) |
| tst2012 | 1.553 | 1.553 | 0 |
| tst2013 | 1.268 | 1.268 | 0 |

1.978 cặp bị loại: 151 rỗng · 1.006 trùng · 760 quá dài · 40 lệch độ dài · 21 rò rỉ.

Tokenizer: fertility 1,0664 (En) và 1,0137 (Vi), tỉ lệ `<unk>` **0,0000%**.

### Các số khác hay bị hỏi

| | |
|---|---|
| KV cache | cùng chuỗi token, nhanh gấp **5,5 lần** (20,3 → 3,7 giây / 40 câu) |
| Phục hồi sau khi giết phiên | lệch **0,0000%** trên 60 bước |
| Chi phí đồng bộ checkpoint | **0,01%** tổng thời gian (yêu cầu < 5%) |
| Kiểm tra kiến trúc | **12/12** đạt, lệch so với PyTorch < 10⁻⁵ |
| Cổng chặn học thuộc 50 câu | loss < 0,05, BLEU **100,00** |
| Cổng chặn với cấu hình A0 | BLEU **100,00** — kiến trúc vanilla không sai |

---

## Câu hỏi khó nhất có thể bị hỏi

> **"12/12 bài kiểm tra đạt có nghĩa mô hình chắc chắn đúng không?"**

**Không.** Kiểm thử chứng minh **sự có mặt của những hành vi đúng cụ thể**, không
chứng minh **sự vắng mặt của mọi lỗi**.

Bằng chứng nằm ngay trong chính đồ án này. Cả 12 bài đều xanh trong khi:

- `dung_kv_cache` bị để `false` suốt — KV cache viết xong mà không ai bật
- `CHI_THI_NGHIEM` bị khai báo **hai lần** trong notebook, dòng dưới đè dòng trên
- warmup 4.000 lớn hơn ngân sách 3.000 bước, giết A0 **ba lần liên tiếp**
- cờ chạy lại xoá bảng kết quả nhưng quên xoá checkpoint, khiến A0 chạy đúng **0
  bước** rồi báo "xong"

**Không lỗi nào trong bốn lỗi trên bị 12 bài test bắt được** — vì chúng là lỗi
**cấu hình** và lỗi **quy trình**, không phải lỗi kiến trúc.

Cách phát biểu đúng:

> "12/12 có nghĩa kiến trúc làm đúng thứ chúng em đặc tả. Nó không có nghĩa cả hệ
> thống đúng — và thực tế đồ án này có bốn lỗi nghiêm trọng lọt qua toàn bộ 12
> bài test."

Trả lời được câu này một cách trung thực sẽ ghi điểm hơn nhiều so với gật đầu.
