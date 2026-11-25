# Tài liệu Định hướng Tổng quan: Trực quan hóa Huấn luyện vs. Suy luận AI & Cơ chế MoE

**Ngày tạo:** 2026-11-25  
**Dự án:** MiniMind 3D Training & Architecture Visualizer (`D:\Projects\ClaudeDemo\AI-Visualize`)  
**Tham chiếu hiện có:** [PLAN.md](../PLAN.md), [2026-11-23-dual-arena-review.md](./reviews/2026-11-23-dual-arena-review.md)

---

## 1. Tuyên ngôn Mục tiêu Cốt lõi (North Star)

Toàn bộ trải nghiệm của ứng dụng xoay quanh **hai câu hỏi trọng tâm nhất** mà người xem cần nắm vững sau khi dùng app:

1. **Khi Huấn luyện (Training):** Chuyện gì thực sự diễn ra bên trong mạng nơ-ron qua từng pha và từng giai đoạn (`Pretrain → Full SFT → LoRA → RL`) để một mô hình từ đoán ngẫu nhiên biến thành biết trả lời có cấu trúc?
2. **Khi Trả về kết quả (Inference / Generation):** Khi người dùng đặt một câu hỏi, mô hình đã học đọc ngữ cảnh, tận dụng bộ nhớ tạm (**KV Cache / GQA**), kích hoạt **MoE** và bốc thăm từng từ tiếp theo như thế nào để nhả ra câu trả lời?

### Nguyên tắc thiết kế & khoa học bắt buộc
- **Trung thực khoa học:** Mọi sơ đồ SVG, nhãn 3D, tỷ lệ tham số, bộ nhớ VRAM và câu khẳng định hiệu năng phải khớp đúng toán học đang chạy trong code (không vẽ khối không tồn tại, không suy diễn kết quả ngoài dữ liệu đo được).
- **Hai mức giải thích song hành:** Chuyển đổi mượt bằng `🌱 Dễ hiểu (Đời thường)` ↔ `🔬 Chuyên sâu (Kỹ thuật)` từ cùng một nguồn dữ liệu.
- **Điểm dừng trực quan (Visual Checkpoints):** Học thông qua hành động *tạm dừng — quan sát — so sánh* tại đúng các thời điểm bước ngoặt thay vì để hoạt ảnh trôi liên tục khó theo kịp.
- **Một tiêu điểm mỗi lúc (Single-Focus Viewport):** Khi mở bảng phân tích chuyên sâu (`🔍 Soi 1 tầng MoE`, `⚔️ Đấu trường Live`), cảnh 3D nền tạm dừng & ẩn hẳn để không gây nhiễu thị giác và tiết kiệm GPU.

---

## 2. Làm rõ Bản chất MoE (Giải đáp các hiểu nhầm phổ biến)

Phần này chốt lại chính xác cách MoE vận hành để toàn bộ Story Mode, Panel X-Ray và Đấu trường Live nói cùng một ngôn ngữ thống nhất:

### 2.1. Số chuyên gia có định nghĩa sẵn từ lúc bắt đầu không?
**Có — cố định 100% ngay từ lúc khai báo kiến trúc (trước Step 0).**
- Trong **MiniMind2-MoE**, lập trình viên khai báo cứng ngay khi tạo mô hình: `4 Routed Experts` (mỗi từ chọn `Top-2`) + `1 Shared Expert` (luôn hoạt động), lặp lại ở cả `8` tầng (`145M` tổng tham số, `45.2M` tham số kích hoạt/từ). Các mô hình thực tế khác cũng cố định trước như Mixtral (`8` experts/tầng) hay DeepSeek-V3 (`256` experts/tầng).
- Mô hình **không tự đẻ thêm hay xóa bớt** khối chuyên gia khi đang train.

### 2.2. Vậy khi nào một "chuyên gia" thực sự hình thành?
Cần phân biệt rạch ròi **"Cái vỏ"** và **"Cái hồn"**:

| Giai đoạn | Trạng thái của Expert & Router | Điều người xem cần thấy trực quan |
| :--- | :--- | :--- |
| **Step 0 (Khởi tạo)** | **Chỉ có 4 "cái vỏ" rỗng.** Cả 4 khối FFN và Router đều chứa số ngẫu nhiên. | Mọi expert đều giống nhau (chỉ là ma trận rỗng ngẫu nhiên), Router chia ~25/25/25/25 một cách vô nghĩa — **chưa có chuyên gia nào**. |
| **Phá vỡ đối xứng (Pretrain sớm)** | Nhờ sai khác nhỏ ở khởi tạo, với một mẫu đầu vào `x`, giả sử `E1` tình cờ giảm Loss tốt hơn `E2` một chút. | Đạo hàm đẩy Router tăng trọng số cho `E1` ở các mẫu giống `x` → `E1` được cập nhật nhiều hơn trên nhóm mẫu đó → vòng lặp tự củng cố *(rich-get-richer)*. |
| **Phân hóa + Cân bằng tải (Pretrain)** | **Aux Loss** (`seq_aux`: $\alpha \cdot \sum ce_i \cdot P_i$, với $\alpha = 0.1$ trong `simulator.ts` và $\alpha = 0.2$ trong phòng thí nghiệm `moeEmergenceCore`/`dualArenaCore`) phạt Router nếu dồn việc lệch. *(Ở DeepSeek-V3 dùng bias động không cần aux loss).* | Cả 4 "vỏ" ban đầu buộc phải tự tách nhau ra đảm nhận các cụm đặc trưng khác nhau trong không gian vector; số chuyên gia hiệu dụng `exp(H(load))` giữ quanh `~4.0/4`. |
| **Tắt Aux Loss (Load Collapse)** | Không còn lực phạt cân bằng → vòng lặp giàu càng giàu chiếm ưu thế. | Với Top-2, tối đa 2 expert ôm hết ~`50/50/0/0`, `exp(H)` tụt từ `4.0 → ~2.0/4`. GPU vẫn trả đủ VRAM cho cả `4` expert (`2,860 MB` ở Pretrain) nhưng dung lượng thật chỉ bằng một nửa. |
| **LoRA & Inference** | LoRA chỉ gắn vào `q_proj`, `o_proj` nên Router và các Expert bị **đóng băng**. Inference không còn cập nhật trọng số. | Vai trò từng expert và cách Router chia việc giữ nguyên trạng thái đã học từ Pretrain/SFT. |

### 2.3. Lưu ý nhất quán giữa dữ liệu thật và dữ liệu thử nghiệm nhỏ
- Trên LLM thật (như mô phỏng trong `🔍 Soi 1 tầng MoE`), Router **không** chia chuyên gia theo nhãn chủ đề con người đặt sẵn ("E0 = địa danh"); nó gom theo cụm ngữ cảnh/cú pháp trừu tượng trong vector ẩn.
- Trong phòng thí nghiệm `③ Chuyên gia tự hình thành` và `⚔️ Đấu trường Live`, màu `Địa lý / Lập trình / Ẩm thực / Toán học` **chỉ là nhãn quan sát bên ngoài** (mô hình không bao giờ thấy nhãn này); do bộ câu thử nghiệm nhỏ có từ vựng tách biệt theo miền nên các cụm tự nhiên vô tình trùng khá sát với màu miền. Mọi màn hình phải nêu rõ sự khác biệt này để không mâu thuẫn nhau.

---

## 3. Bảng đối chiếu 1–1: Huấn luyện (Training) vs. Trả về kết quả (Inference)

Sự nhầm lẫn lớn nhất của người mới là nghĩ AI "đọc đề rồi trả lời" giống hệt lúc nó "học". Bảng dưới đây là xương sống để app trực quan hóa hai chế độ tách biệt:

| Khía cạnh | 🏋️ Huấn luyện (Training — đang có) | 💬 Trả về kết quả (Inference — cần bổ sung rõ) |
| :--- | :--- | :--- |
| **Luồng đi của câu** | Đưa **cả chuỗi song song** (`B × L`) qua `L0 → L7` trong một lượt. | Chia 2 pha: **Prefill** (đọc cả câu hỏi 1 lần) → **Decode** (sinh **từng từ một** nối đuôi đến `<eos>`). |
| **Nhịp trong 1 bước** | `1. Forward → 2. Loss → 3. Backward`. | `1. Prefill (tạo KV Cache) → 2. Forward lấy logits → 3. Softmax + Sampling → 4. Nối từ & lặp`. |
| **Loss & Backward** | **Bắt buộc có.** Sóng đỏ/tím quét ngược từ `L7 → L0` để chỉnh trọng số. | **Hoàn toàn không có.** Trọng số đóng băng 100%, không tính đạo hàm, không cần AdamW. |
| **Vai trò của GQA (`8Q : 2KV`)** | Ít ảnh hưởng VRAM train (vẫn phải lưu kích hoạt cả chuỗi cho Backward). | **Tỏa sáng tại đây:** Giảm `75%` dung lượng **KV Cache** tích lũy dần theo độ dài câu trả lời. |
| **Vai trò của MoE (`45.2M / 145M`)** | Cần VRAM cho đủ `145M` trọng số + Grads + AdamW (`2,860 MB` ở Pretrain). | Mỗi từ sinh ra chỉ kích hoạt `45.2M / 145M` tham số (~1/3.2 công sức so với Dense `145M`), tăng tốc độ sinh từ. |
| **Vai trò của RoPE** | Gắn góc xoay $m \cdot \theta$ cho cả `L` vị trí song song một lúc. | Prefill gắn vị trí `0..m-1`; mỗi bước Decode xoay đúng kim thêm 1 nấc `m, m+1, m+2...`. |
| **Yếu tố điều khiển đầu ra** | Learning Rate, Loss Masking (SFT), LoRA Rank-8, DPO KL / GRPO Advantage. | **Temperature**, Top-k / Top-p (quyết định chọn từ an toàn nhất hay đa dạng hơn). |

---

## 4. Hệ thống "Các Điểm Dừng Trực Quan" (Visual Checkpoints)

Thay vì chỉ bấm Pause ngẫu nhiên, người xem cần đi qua các **điểm dừng có ý nghĩa** rõ ràng:

### Nhóm A: Điểm dừng trong Quá trình Huấn luyện (`Training`)
1. **Điểm dừng Pha 1 — Đọc song song (Forward):** Cả 10 hạt từ đi ngang hàng qua `L0 → L7`, chứng minh máy không đọc từng chữ một khi học.
2. **Điểm dừng Pha 2 — Chấm Loss & Prompt Masking (Pretrain vs SFT):** Tại cổng LM Head, thấy rõ Pretrain chấm mọi từ, còn SFT che phần câu hỏi của `User` (`ignore_index = -100`, màu xám) chỉ chấm phần `Assistant`.
3. **Điểm dừng Pha 3 — Sóng Backward & LoRA 🔒:** Thấy sóng đỏ vặn 100% mạng ở Pretrain/SFT, đối lập với LoRA nơi mạng gốc đóng băng (sóng mờ) và chỉ nhánh tím $A \times B$ cập nhật (`0.51%` Dense / `0.11%` MoE) làm thanh VRAM AdamW sụt mạnh.
4. **Điểm dừng Chuỗi 4 Giai đoạn (`Pretrain → SFT → LoRA → DPO/GRPO`):** Click từng giai đoạn để thấy câu trả lời cho *cùng 1 câu hỏi* tiến hóa từ "nói tiếp văn thô" → "trả lời đúng khuôn" → "nhẹ VRAM" → "lịch sự, hợp ý người dùng".

### Nhóm B: Điểm dừng trong Quá trình Trả về Kết quả (`Inference` — mới)
1. **Điểm dừng Prefill:** Toàn bộ prompt bay 1 lần qua mạng, sinh ra **tủ hồ sơ KV Cache** ban đầu (`m` cột đầu tiên).
2. **Điểm dừng Xác suất & Temperature:** Dừng tại cổng LM Head trước khi chốt từ: xem Top-5 từ ứng viên, kéo thanh **Temperature** (`0.0` chọn chắc chắn từ top 1 vs `1.0` mở rộng lựa chọn).
3. **Điểm dừng Decode từng bước (`+1 Từ tiếp theo`):** Chỉ **1 hạt từ mới nhất** đi xuyên qua `L0 → L7`, **chỉ đọc lại KV Cache đã lưu** thay vì tính lại từ đầu; thấy kim RoPE nhích thêm 1 nấc, 2 chuyên gia MoE sáng lên cho riêng từ đó, rồi từ vừa chọn bay về đầu chuỗi để chuẩn bị bước kế tiếp.
4. **Điểm dừng Kết thúc (`<eos>`):** Thấy câu trả lời hoàn chỉnh và tổng kết KV Cache đã tiết kiệm nhờ GQA `8Q:2KV` (so với MHA `8KV`).

### Nhóm C: Điểm dừng Riêng cho MoE ("Vỏ có sẵn → Chuyên gia tự hình thành")
1. **Điểm dừng Step 0:** Khẳng định *"Số lượng `4` chuyên gia đã cố định từ trước khi train; lúc này cả 4 chỉ là vỏ rỗng ngẫu nhiên."*
2. **Điểm dừng Phá vỡ đối xứng & Phân hóa:** Tua ở Epoch 1 → Epoch 15 (trong Tab ③ hoặc Đấu trường) để xem tỷ lệ phân bổ tự lệch dần từ ngẫu nhiên sang chuyên môn hóa mà không cần gán nhãn.
3. **Điểm dừng So sánh có / không Aux Loss & Cột 3 (`Dense 18` vs `MoE 18/32` vs `Dense 32`):** Thấy rõ cả cái giá bộ nhớ (`1.05×` trong arena nhỏ, `2,860 MB` vs `734 MB` ở MiniMind2) lẫn rủi ro sụp đổ tải (`exp(H) → ~2.0/4`) nếu bỏ cân bằng tải.

---

## 5. Đánh giá Hiện trạng App & Khoảng trống UX

| Thành phần hiện có | Đường dẫn nguồn | Trạng thái & Đánh giá |
| :--- | :--- | :--- |
| Scene 3D 8 tầng + 3 pha + LoRA + DPO/GRPO + Suspension | [MiniMindScene.ts](../packages/client/src/scene/MiniMindScene.ts) | **Rất tốt** cho Training; chưa có trạng thái **Inference Mode** (ẩn Loss/Backward, vẽ Prefill + 1-token Decode + thanh KV Cache tăng dần). |
| Bộ điều phối chính & HUD 2 cột + Story Mode 8 cảnh | [main.ts](../packages/client/src/main.ts), [index.html](../packages/client/index.html) | Topbar đang dàn ngang quá nhiều nút (~20 nút cùng cấp); Story Mode mới dừng ở cảnh 8 (chưa nối sang Inference). |
| Simulator dùng chung & VRAM Breakdown 5 đoạn | [simulator.ts](../packages/shared/src/simulator.ts) | Chuẩn thông số MiniMind2 (`25.8M` Dense, `145M/45.2M` MoE); đã dùng `exp(H)` nhất quán. Cần bổ sung bước mô phỏng `InferenceFrame` (Prefill/Decode, KV Cache MB, Temperature sampling). |
| Panel `🔍 Soi 1 tầng MoE` (3 tabs) | [moeXray.ts](../packages/client/src/ui/moeXray.ts), [moeEmergenceCore.ts](../packages/client/src/workers/moeEmergenceCore.ts) | Đã cố định header, ẩn 3D nền khi mở; nên bổ sung nhãn rõ ngay đầu panel rằng `4` chuyên gia được khai báo cố định từ Step 0. |
| Modal `⚔️ Đấu trường Live` (`Dense 18` / `MoE 18/32` / `Dense 32`) | [dualArena.ts](../packages/client/src/ui/dualArena.ts), [dualArenaCore.ts](../packages/client/src/workers/dualArenaCore.ts) | Đã kiểm chứng trung thực qua 5 seed và khớp sơ đồ SVG thật (`Trộn ngữ cảnh`). |
| Micro-LM Web Worker | [microLmWorker.ts](../packages/client/src/workers/microLmWorker.ts) | Đã chạy Backprop thật trên trình duyệt; có thể dùng làm động cơ sinh từ thật cho chế độ Inference thử nghiệm khi không gắn Python backend. |

### 3 Khoảng trống lớn nhất cần giải quyết
1. **Quá tải nút ở màn hình đầu:** Người mới vào chưa biết bắt đầu từ đâu vì nút kiến trúc, giai đoạn, Story Mode, X-Ray, Đấu trường Live, Micro-LM, speed, step đứng ngang hàng nhau.
2. **Thiếu hẳn nửa sau câu chuyện — "Cách AI trả về kết quả (Inference)":** GQA (`8Q:2KV`), Temperature, Autoregressive Decode, KV Cache hiện mới chỉ được *kể bằng chữ*, chưa *chạy trực quan từng bước* để người dùng bấm dừng/xem.
3. **Mạch MoE Step 0 → Hình thành chưa thành một luồng dẫn tự nhiên:** Thông điệp *"Số chuyên gia cố định từ Step 0, chuyên môn hóa mới hình thành qua dữ liệu"* cần xuất hiện đúng điểm vào Story Mode cảnh 4 và X-Ray, trước khi người dùng đi vào đồ thị phức tạp.

---

## 6. Lộ trình Tái cấu trúc đề xuất (Đầu vào cho Implementation Plan)

### Giai đoạn 1 — Tổ chức lại Kiến trúc Thông tin & Làm rõ MoE từ Step 0
- **Phân cấp Topbar thành 2 tầng rõ ràng:**
  - **Trục chính ở giữa/trái:** Công tắc chế độ lớn **`🏋️ 1. Quá trình Huấn luyện (Training)` ↔ `💬 2. Trả về kết quả (Inference)`**, bên cạnh `Dense ↔ MoE`, `Story Mode`, và `🌱 Dễ hiểu ↔ 🔬 Chuyên sâu`.
  - **Nhóm `🧪 Phòng thí nghiệm (Labs)`:** Gom `🔍 Soi 1 tầng MoE`, `⚔️ Đấu trường Dense vs MoE`, và `🧪 Micro-LM Web Worker` thành nhóm phụ gọn (hoặc menu/thanh phụ) để không lấn át luồng chính.
- **Nhấn mạnh MoE Step 0:** Bổ sung thẻ tóm tắt 1 dòng trong Story Cảnh 4 và Tab ① / ③ của X-Ray: *"Khởi tạo sẵn 4 vỏ chuyên gia rỗng (Step 0) → Học qua dữ liệu mới phân vai (không tự thêm bớt expert, không gán chủ đề sẵn)"*, kèm nút nhảy nhanh về mốc `Step 0` vs `Đã hội tụ`.

### Giai đoạn 2 — Chế độ Trực quan hóa Trả về Kết quả (`Inference Mode`)
- **Động cơ Inference (`shared/simulator.ts` + nối tùy chọn `Micro-LM` / `minimind_infer_server.py`):**
  - Cho chọn câu hỏi mẫu hoặc nhập prompt ngắn; chọn checkpoint giả lập theo giai đoạn (`Sau Pretrain`, `Sau SFT`, `Sau LoRA`, `Sau DPO/GRPO`) để thấy ngay sự khác biệt khi trả lời.
  - Chỉnh **Temperature** (`0.0 – 1.2`) xem bảng xác suất Top-5 biến đổi trực tiếp.
- **Điều khiển điểm dừng Inference:**
  - Nút **`▶ Đọc câu hỏi (Prefill)`** → dừng ngay sau khi nạp KV Cache ban đầu.
  - Nút **`⏭ Sinh 1 từ tiếp theo`** (hoặc **`▶ Tự sinh từng từ`**) → chạy đúng 1 bước Decode rồi dừng tại điểm chốt từ.
- **Hình ảnh 3D trong `MiniMindScene.ts`:**
  - Ẩn vòng `Loss Burst` và sóng đỏ `Backward` (hiển thị badge rõ: *"Trọng số đóng băng — Không có Loss / Backward"*).
  - Pha Prefill: cả nhóm hạt câu hỏi đi qua `L0 → L7`, thắp sáng dãy khối **KV Cache (`2KV`)** ở mỗi tầng.
  - Pha Decode: chỉ **1 hạt từ mới nhất** đi xuyên qua mạng, vẽ đường đọc lại KV Cache cũ, xoay kim **RoPE** đúng nấc vị trí `m`, sáng cặp Top-2 Expert của riêng từ đó (nếu đang bật MoE), sau đó nối từ mới vào câu trả lời ở HUD.
- **Thanh đo KV Cache sống (GQA 8Q:2KV vs MHA 8KV):** Hiển thị số token đã cache tăng lên sau mỗi bước Decode và mức tiết kiệm `75%` bộ nhớ nhờ GQA.

### Giai đoạn 3 — Hợp nhất Story Mode (10 cảnh liền mạch) & Kiểm chứng
- Mở rộng **Story Mode** từ 8 cảnh thành **10 cảnh trọn vòng đời AI**:
  - **Phần I — Huấn luyện (Cảnh 1–8):** Giữ 8 cảnh hiện có, làm rõ **Cảnh 4** bắt đầu từ *"4 vỏ chuyên gia cố định trước Step 0"* trước khi vào rẽ nhánh.
  - **Phần II — Trả về kết quả (Cảnh 9–10):** Cảnh 9 (*Prefill & KV Cache / GQA khi nhận câu hỏi*) → Cảnh 10 (*Vòng lặp Decode từng từ, Temperature & MoE thực thi nhanh*).
- Kiểm chứng lại `npm run build` sạch và test đầy đủ trên trình duyệt ở cả `1280×720` và `1440×900` (không chồng lấp UI, 0 lỗi console).

---

## 7. Các quyết định cần chốt trước khi viết Implementation Plan chi tiết

1. **Cách hiển thị Inference Mode:** Dựng trực tiếp trên **cùng sân khấu 3D chính + HUD 2 cột** (chuyển mượt bằng công tắc `🏋️ Huấn luyện ↔ 💬 Trả về kết quả`), hay làm một modal riêng như Đấu trường? *(Khuyến nghị: trên sân khấu 3D chính để tận dụng trực tiếp các khối RoPE, GQA/KV Cache và Top-2 Expert đã dựng sẵn).*
2. **Thông số kiến trúc tham chiếu:** Tiếp tục giữ **MiniMind2** (`Dense 25.8M`, `MoE 145M/45.2M Top-2 + Shared`, `GQA 8Q:2KV`) hay lên kế hoạch chuyển sang **MiniMind-3** (`MoE Top-1` không Shared)? *(Khuyến nghị: giữ MiniMind2 cho đợt tái cấu trúc này để tập trung dứt điểm trục Training ↔ Inference và không phải viết lại toàn bộ sơ đồ MoE).*
3. **Thứ tự Git:** Có **commit & push** ngay các bản sửa dừng/ẩn 3D nền + sửa layout X-Ray hiện đang nằm trên nhánh `docs/dual-arena-review` trước khi mở nhánh mới thực hiện Lộ trình này không?
