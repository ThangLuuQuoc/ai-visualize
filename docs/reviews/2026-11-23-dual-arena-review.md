# Review: Đấu trường Train Song song Dense vs MoE (Live)

- **Ngày:** 2026-11-23
- **Commit được review:** `7a3c8c2` — *feat: add real MoE emergence lab (Tab 3), side-by-side Iso-FLOPs Dense vs MoE Live Arena…*
- **Phạm vi:** chỉ phần Đấu trường (`⚔️ Dense vs MoE (Live)`):
  - `packages/client/src/workers/dualArenaCore.ts`
  - `packages/client/src/workers/dualArenaWorker.ts`
  - `packages/client/src/ui/dualArena.ts`
  - phần tích hợp trong `main.ts` / `index.html`

  Tab ③ "Chuyên gia tự hình thành" (`moeEmergenceCore.ts`) và layout X-Ray 1440×900 chưa được review.

## Tóm tắt

Phần kỹ thuật chạy tốt. Vấn đề nằm ở **thông điệp mà người xem nhận được**:

- MoE thua Dense ở mọi seed đã thử, trong khi giao diện gọi đây là "so sánh công bằng".
- Sơ đồ vẽ những khối mà mô hình không hề có.
- Một số con số và mô tả trên giao diện không khớp với code.

Cần sửa nhóm 1–4 trước khi dùng Đấu trường để trình bày.

## Đã kiểm chứng ổn

- **Backprop đúng:**
  - Gradient router đi qua softmax có nhiệt độ: `ds = p·(dP − Σp·dP)/T`.
  - Gradient aux theo `α·Σ ceᵢ·Pᵢ` (coi `ceᵢ` là hằng).
  - Gradient expert được nhân với `gA`/`gB`.
  - Gradient embedding được chia đúng tỉ lệ 0.4 / 0.6.
- Hai mạng nhận **cùng một mini-batch** ở mỗi bước.
- Snapshot từng Epoch, thanh tua lịch sử và worker tách riêng đều hoạt động.
- `npm run build` sạch, không lỗi TypeScript. Vite tách `dualArenaWorker` thành chunk riêng (21.84 kB).
- Kiểm tra trên trình duyệt (`localhost:5273`): chạy từng Epoch, đổi từ kiểm tra và tua lịch sử đều không có lỗi console.
- Đổi selector sang `#arch-switcher button[data-arch]` là đúng: các nút X-Ray và Đấu trường không còn kích hoạt nhầm `SET_ARCH`.

## Số liệu đo lại (headless, `npx tsx`)

Kích thước dữ liệu và mô hình:

| | Giá trị |
|---|---|
| Từ vựng `V` | 366 |
| Số mẫu (token) trong dữ liệu | 880 |
| Số mẫu mỗi "Epoch" | 768 (12 batch × 64) |
| Tổng tham số Dense / MoE | 18,840 / 19,722 (**1.05×**) |
| Tham số FFN Dense / MoE | 906 / 1,788 (1.97×) |

CE loss sau 15 Epoch:

| Seed | 1 | 7 | 42 | 123 | 999 |
|---|---|---|---|---|---|
| Dense | 0.167 | 0.137 | 0.153 | 0.161 | 0.133 |
| MoE, aux BẬT | 0.295 | 0.239 | 0.313 | 0.277 | 0.238 |
| MoE, aux TẮT | 0.310 | 0.382 | 0.355 | 0.351 | 0.268 |
| MoE, aux BẬT, `SHARED_SCALE = 1` | 0.194 | 0.187 | 0.286 | 0.227 | 0.185 |

- Số chuyên gia hiệu dụng khi bật aux là 3.21–3.98/4, khi tắt aux là 1.77–2.47/4. Phần này thể hiện đúng hiện tượng sụp đổ tải.
- Chạy 25 Epoch (seed 1, 7, 42): Dense ≈ 0.03, MoE ≈ 0.06–0.09. MoE vẫn thua.
- Câu sinh ở seed 42, Epoch 15:
  - Dense: "Hà Nội là thủ đô ở **miền Bắc** và"
  - MoE: "Hà Nội là thủ đô ở **miền Nam** có"

## Vấn đề (theo mức độ nặng)

### 1. MoE thua Dense ở mọi seed nhưng vẫn được gọi là "so sánh công bằng"

MoE thua ở 5/5 seed, cả khi bật lẫn tắt aux. Người xem sẽ rút ra kết luận "MoE kém hơn".

Dòng giải thích "lợi thế MoE bùng nổ khi dữ liệu lớn gấp nghìn lần" là khẳng định không có gì trong app chứng minh.

Nguyên nhân trong `dualArenaCore.ts`:

- **`SHARED_SCALE = 0.2`:** Shared expert bị thu nhỏ 5 lần, trong khi MiniMind2 cộng Shared với trọng số 1. Đặt lại bằng 1 thì khoảng cách thu hẹp rõ, nhưng MoE vẫn thua (bảng trên).
- **Trọng số gate không được chuẩn hóa:**
  - Khi chạy, mạng dùng xác suất gate thô `gA`, `gB` có tổng nhỏ hơn 1, cộng thêm `ROUTER_TEMP = 0.48`.
  - UI lại hiển thị trọng số đã chuẩn hóa (ví dụ ×0.62).
  - Hai con số không khớp nhau và không giống `norm_topk_prob=True` của MiniMind2.
- **Khởi tạo khác nhau:** Dense dùng seed `seed + 101`, MoE dùng `seed`. Embedding và lớp đầu ra của hai mạng không xuất phát từ cùng một điểm, nên chênh lệch một phần đến từ may rủi khởi tạo.
- **FFN không phải chỗ thắt nút:** embedding và lớp đầu ra chiếm 17,568 trên khoảng 19k tham số, nên thêm dung lượng FFN không giúp được nhiều.

### 2. Con số "bộ nhớ" sai và không nhất quán

- Bản Dễ hiểu ghi "bộ nhớ ~1.8×", bản Chuyên sâu ghi "~1.97×". Cả hai chỉ tính phần FFN.
- Tổng tham số thực tế chỉ chênh **1.05×**.

### 3. Sơ đồ SVG vẽ khối không tồn tại

- SVG có khối **"Attention (Giống hệt MoE)"** và mũi tên **residual** vòng qua FFN.
- Lõi thật không có attention, chỉ trộn ngữ cảnh bằng `0.4·emb(từ hiện tại) + 0.6·trung bình(emb ngữ cảnh)`.
- Lõi cũng không có residual: đầu ra là `MLP(x)` đưa thẳng vào `Wout`.
- Comment trong code ghi "SwiGLU FFN", nhưng thực tế là MLP dùng `tanh`.

### 4. Định nghĩa "Epoch" sai

- Tooltip ghi "Train thêm đúng 1 lượt đọc hết 64 câu".
- Thực tế mỗi Epoch chỉ có 768 mẫu, trong khi dữ liệu có 880 token, nên đọc được khoảng 87%. Ranh giới Epoch trôi dần sau mỗi lượt.
- Dữ liệu không được xáo trộn.

### 5. Bật/tắt Aux giữa chừng làm lẫn lịch sử

`SET_AUX_LOSS` không khởi tạo lại mô hình và không ghi trạng thái aux vào snapshot. Biểu đồ và thanh tua trộn Epoch có aux với Epoch không có aux mà người xem không phân biệt được.

Hướng sửa (chọn một):

- Đổi aux thì reset mô hình, có báo trước cho người xem.
- Hoặc lưu `auxLossEnabled` vào từng snapshot và đánh dấu trên biểu đồ.

### 6. Mâu thuẫn với panel X-Ray

- Panel X-Ray nói router thật không chia việc theo chủ đề con người đặt tên được.
- Đấu trường lại gắn nhãn "E1 Lập trình 46%".
- Trên dữ liệu đồ chơi này, mỗi chủ đề dùng bộ từ vựng gần như tách biệt nên hiện tượng đó có thật. Cần thêm một dòng nói rõ: đây là hiệu ứng của dữ liệu nhỏ có từ vựng tách biệt; ở LLM thật, chuyên môn hóa thường diễn ra ở mức token hoặc cú pháp.

### 7. Lỗi nhỏ

- **Câu viết tiếp lặp lại prompt.** `generate()` trả về cả prompt, còn UI cũng in prompt ở phía trước, ra `“Python là” → Python là là món ăn…`.
- **"Chuyên gia hiệu dụng" dùng hai công thức khác nhau.** Ở đây tính bằng `1/Σl²`, còn simulator và X-Ray dùng `exp(H)`.
- **Khi tắt aux, ô Aux hiện 0.** Lý do là giá trị bị nhân với `α = 0`, nên che mất mức mất cân bằng. Nên hiện giá trị chưa nhân `Σ ceᵢ·Pᵢ`.
- **`auxLoss` trong snapshot lấy từ batch train cuối.** Nó không được tính lại lúc đánh giá; ở Epoch 0 là hằng 0.2.
- **Biểu đồ khó đọc.** Loss và % phân hóa dùng chung một trục mà không có thang chia, và chỉ có loss của MoE được ghi số.
- **Bố cục màn hẹp.** Dưới khoảng 1200px, hai cột xếp chồng lên nhau nên mất ý "đặt cạnh nhau".
- **`setInterval` trong worker không bao giờ được xóa.** Khi tạm dừng nó vẫn chạy mỗi 550ms (vô hại nhưng thừa).
- **Commit gộp 3 tính năng khác nhau.** Commit `7a3c8c2` gồm Đấu trường, Tab ③ và layout X-Ray, nên khó review và khó revert riêng từng phần.

## Đề xuất sửa (thứ tự ưu tiên)

1. **Cho so sánh thật sự công bằng:**
   - Đặt `SHARED_SCALE = 1`.
   - Chuẩn hóa trọng số Top-2 khi chạy, kèm gradient qua bước chuẩn hóa.
   - Cho Dense và MoE dùng chung khởi tạo `E`, `Wout`, `bout`.
2. **Sửa con số bộ nhớ:** hiện cả tổng tham số (1.05×) và tham số FFN (1.97×), có ghi rõ từng loại.
3. **Sửa SVG:** đổi khối "Attention" thành "Trộn ngữ cảnh (trung bình embedding)" và bỏ mũi tên residual. Hoặc thêm residual thật vào cả hai lõi.
4. **Định nghĩa lại Epoch:** mỗi Epoch đọc đủ 880 mẫu, có xáo trộn bằng PRNG theo seed.
5. **Đo lại sau khi sửa 1–4.** Nếu MoE vẫn thua khi cùng số nơ-ron tính mỗi từ, ghi thẳng điều đó trên giao diện.
6. **Thêm cột thứ ba "Dense 32 nơ-ron (cùng tổng tham số)",** giống thẻ so sánh 3 cột `archCompare.ts`. Lợi thế thật cần chứng minh của MoE là *chất lượng gần bằng mạng to nhưng chi phí tính mỗi từ bằng mạng nhỏ*.
7. Xử lý các mục 5–7 ở trên.
