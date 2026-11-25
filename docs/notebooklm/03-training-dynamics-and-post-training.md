# Toàn cảnh Huấn luyện & Tinh chỉnh LLM — Pretrain, SFT, LoRA, DPO & GRPO

> Tài liệu này giải phẫu đầy đủ điều gì xảy ra bên trong mô hình từ lúc tính toán 1 bước học (`Forward → Loss → Backward → AdamW`), phân rã bộ nhớ VRAM thực tế, **Scaling Laws (Chinchilla)**, tới chuỗi tinh chỉnh 4 giai đoạn: **Pretrain → Full SFT → LoRA → RL (DPO / GRPO)**.

---

## 1. Giải phẫu một bước Huấn luyện (Training Step)

Khi huấn luyện mô hình ngôn ngữ nhân quả (Causal LM), một batch có kích thước $[B, L]$ (ví dụ `L = 10` từ trong mô phỏng) đi qua đúng 3 pha:

### Pha 1 — Forward ("Đọc cả câu song song")
- Khác hoàn toàn lúc trả lời cho người dùng (sinh từng chữ), khi **train** mô hình biết sẵn cả câu đáp án, nên toàn bộ $L$ token đi qua `L0 → L_{N-1}` **cùng một lúc** dưới dạng phép nhân ma trận lớn (`[B, L, d]`).
- Mặt nạ tam giác (**Causal Mask**) đảm bảo tại vị trí $t$, token đó chỉ được nhìn thấy `0..t`, không nhìn trộm `t+1..L-1`. Vì vậy một lần Forward sinh ra đồng thời $L$ dự đoán kế tiếp!

### Pha 2 — Loss ("Chấm độ bất ngờ")
- Tại cổng đầu ra (`LM Head`), với mỗi vị trí $t$ có phân phối xác suất dự đoán $P(\cdot \mid w_{\le t})$ và chữ đúng thực tế là $w_{t+1}$, độ sai lệch đo bằng **Cross-Entropy Loss**:
  $$\mathcal{L}_t = -\ln P(w_{t+1} \mid w_{\le t})$$
- **Trực giác "Độ bất ngờ" & Perplexity:**
  - Nếu mô hình đoán chữ đúng với xác suất $P = 96\% \Rightarrow -\ln(0.96) \approx 0.04$ (rất ít bất ngờ).
  - Nếu mô hình đoán ngẫu nhiên trên từ điển $V = 6,400$: $P = \frac{1}{6400} \Rightarrow \mathcal{L} = \ln(6400) \approx 8.76$.
  - **Perplexity** chính là $\text{PPL} = e^{\mathcal{L}}$ — hiểu nôm na là "mô hình đang phân vân giữa trung bình bao nhiêu từ có xác suất ngang nhau".
- **Tách biệt Loss từng từ vs. Loss cả sách (Batch Average):** Đôi khi mô hình tự tin một từ mẫu quen thuộc (`Loss_token = 0.05`) trong khi Loss trung bình toàn batch vẫn cao (`4.48`) vì còn nhiều ngữ cảnh khác đang học.

### Pha 3 — Backward & Cập nhật AdamW ("Vặn núm")
- Thuật toán Lan truyền ngược (**Backpropagation**) quét đạo hàm ngược từ cổng Loss về `L7 → L0 → Embedding`.
- Bộ tối ưu **AdamW** cập nhật từng trọng số dựa trên gradient hiện tại kết hợp trung bình trượt bậc 1 (**momentum $m$**) và bậc 2 (**variance $v$**) cộng trừ suy giảm trọng số (*weight decay*).

---

## 2. Phân rã Bộ nhớ VRAM khi Train (Vì sao 1B model không thể train với 2GB VRAM?)

Khi train bằng Mixed Precision **bf16 + AdamW**, mỗi tham số đang được huấn luyện tốn tới **`16 bytes`** trạng thái tĩnh (chưa kể Activations!):

| Thành phần VRAM | Kích thước / tham số | Ví dụ MiniMind2-Dense (`25.8M`) Pretrain | Ví dụ MiniMind2-MoE (`145M`) Pretrain | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :---: | :--- |
| **1. Trọng số mô hình (`Weights`)** | `2 B/param` (bf16) | `~52 MB` | `~290 MB` | Toàn bộ tham số của mạng |
| **2. Đạo hàm (`Gradients`)** | `2 B/param` train | `~52 MB` | `~290 MB` | Lưu $\nabla W$ cho các lớp mở khóa |
| **3. Trạng thái AdamW** | **`12 B/param` train** | `~310 MB` | **`~1,740 MB`** | **FP32 Master (`4B`) + Momentum `m` (`4B`) + Variance `v` (`4B`)** |
| **4. Bản gốc tham chiếu $\pi_{\text{ref}}$** | `2 B/param` | `0` (chỉ có ở DPO/GRPO: `~52 MB`) | `0` (`~290 MB` ở DPO/GRPO) | Mô hình đóng băng để tính phạt KL |
| **5. Kích hoạt (`Activations`)** | Phụ thuộc $B \times L \times d$ | `~320 MB` | `~540 MB` | Giữ đầu ra trung gian từ Forward để dùng cho Backward |
| **Tổng VRAM ước tính** | — | **`~734 MB`** | **`~2,860 MB`** | **AdamW 12B là kẻ ngốn VRAM lớn nhất khi full pretrain** |

**Hai mẹo cứu VRAM thực chiến:**
- **Activation Checkpointing (Gradient Checkpointing):** Không lưu toàn bộ activations trung gian trong lúc Forward; khi Backward cần tới tầng nào thì chạy tính lại Forward cục bộ cho tầng đó — giảm mạnh VRAM Activation (đổi thêm ~25–33% thời gian tính).
- **Gradient Accumulation:** Chia batch to thành nhiều micro-batch nhỏ, cộng dồn `.grad` qua nhiều lượt rồi mới gọi `optimizer.step()` một lần.

---

## 3. Scaling Laws (Kaplan & Chinchilla) — Vì sao dữ liệu nhỏ chưa thấy sức mạnh MoE?

- **Chinchilla Scaling Law (Hoffmann et al., 2022):** Để đạt chất lượng tối ưu trên một ngân sách FLOPs cố định $C \approx 6 N D$, kích thước mô hình $N$ và số token huấn luyện $D$ cần tăng cân xứng theo tỷ lệ vàng:
  $$D \approx 20 \cdot N$$
  (Ví dụ mô hình `145M` tham số lý tưởng cần tối thiểu ~`2.9 tỷ` token đọc qua; mô hình `70B` cần ~`1.4 nghìn tỷ` token).
- **Ý nghĩa cho MoE vs Dense:**
  - Trên **dữ liệu đồ chơi cực nhỏ** (vài trăm hoặc vài nghìn câu), cả mạng Dense nhỏ lẫn MoE đều nhanh chóng *học thuộc lòng (overfit)*; thậm chí MoE còn chia bước cập nhật của mỗi expert đi (~$K/N$ lượt so với Dense) và phần lớn tham số lại nằm ở `Embedding + LM Head` chứ không phải FFN.
  - Chỉ khi bước vào vùng **hàng tỷ – nghìn tỷ token**, một khối Dense kích hoạt tương đương bị **quá tải dung lượng ghi nhớ (capacity bottleneck)** ở FFN, lúc đó tổng sức chứa $N_{\text{total}}$ lớn gấp nhiều lần của MoE mới tạo cách biệt rõ rệt về Loss/Perplexity.

---

## 4. Chuỗi 4 Giai đoạn Huấn luyện (`Pretrain → SFT → LoRA → RL`)

### Giai đoạn 1 — Pretrain (Tự giám sát trên văn bản thô)
- **Mục tiêu:** Học ngữ pháp, cấu trúc lập luận và kiến thức thế giới bằng cách đoán tiếp từ trên hàng tỷ–nghìn tỷ token.
- **Đặc điểm:** Mở khóa **100%** tham số, chấm Loss trên mọi token. Kết quả sau Pretrain là mô hình chỉ biết **viết tiếp đoạn văn** (Completion), chưa biết đối đáp theo vai `User / Assistant`.

### Giai đoạn 2 — Full SFT (Supervised Fine-Tuning & Prompt Loss Masking)
- **Mục tiêu:** Dạy mô hình khuôn mẫu Hỏi–Đáp (`<|user|> ... <|assistant|> ...`).
- **Cơ chế quan trọng — Prompt Loss Masking (`ignore_index = -100`):**
  Cả câu hỏi `User` và câu trả lời `Assistant` cùng đi vào Forward. Tại LM Head, các vị trí thuộc phần `User` được gắn nhãn `-100` để **không tính Cross-Entropy Loss** (mô hình không bị phạt vì đoán câu hỏi của người dùng).
  **Lưu ý:** Dù không phát sinh Loss ở các vị trí `User`, sóng Backward từ phần `Assistant` **vẫn truyền ngược qua các vector K, V của phần `User`** vì Attention của câu trả lời phụ thuộc vào ngữ cảnh câu hỏi!
- **Rủi ro:** Cập nhật toàn bộ mạng bằng tập SFT nhỏ dễ gây **Catastrophic Forgetting** (quên kiến thức nền đã học ở Pretrain).

### Giai đoạn 3 — LoRA & QLoRA (Low-Rank Adaptation)
- **Ý tưởng toán học:** Thay vì mở khóa ma trận gốc khổng lồ $W_0 \in \mathbb{R}^{d \times d}$, ta **đóng băng $W_0$** và chỉ gắn thêm hai ma trận gầy hạng thấp $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$ ($r \ll d$, ví dụ $r = 8$):
  $$W = W_0 + \frac{\alpha}{r} B A$$
  Khi khởi tạo: $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ để bước đầu không làm lệch đầu ra cũ của mô hình.
- **Số liệu thực tế trên MiniMind2 (gắn vào `q_proj`, `o_proj`):**
  - Dense `25.8M`: chỉ train **`0.13M`** (`0.51%`).
  - MoE `145M`: chỉ train **`0.16M`** (`0.11%`). *(Router và các Expert đóng băng hoàn toàn).*
- **Vì sao LoRA tiết kiệm VRAM cực mạnh nhưng tốc độ chỉ nhanh hơn ~12–15%?**
  - **Tiết kiệm VRAM (>50–70%):** Gần như xóa sổ phần **`12 bytes/param` của AdamW** và `2 bytes/param` của Gradients cho $99.5\%$ trọng số gốc (cắt vùng AdamW ở MoE từ `1,740 MB` xuống chỉ vài MB). Kết hợp lượng tử hóa $W_0$ xuống **4-bit NF4** ta có **QLoRA**.
  - **Không tăng tốc gấp nhiều lần:** Bởi vì khi **Backward**, đạo hàm kích hoạt $\frac{\partial \mathcal{L}}{\partial x}$ **vẫn phải chạy ngược xuyên qua toàn bộ $W_0$ bị khóa** để tới được các nhánh $A \times B$ ở tầng dưới (*hiệu ứng 2 sóng: sóng mờ qua mạng gốc + sóng tím cập nhật $A, B$*).

### Giai đoạn 4 — RL / Alignment (PPO → DPO → GRPO)
Sau SFT, mô hình đã biết trả lời nhưng chưa phân biệt câu trả lời hay vs. dở, thật vs. bịa, an toàn vs. độc hại. Ta cần căn chỉnh hành vi (Alignment), bắt đầu từ mô hình SFT:

1. **RLHF với PPO (Cách cổ điển):** Cần nuôi cùng lúc **4 mô hình** trên GPU (`Policy đang học`, `Reference đóng băng`, `Reward Model`, và `Critic/Value Network`) → cực kỳ tốn VRAM và khó ổn định hyperparameter.
2. **DPO (Direct Preference Optimization):**
   - Dùng một biến đổi giải tích chứng minh hàm thưởng ẩn có thể viết trực tiếp qua tỷ số xác suất giữa Policy $\pi_\theta$ và Reference $\pi_{\text{ref}}$ trên cặp câu trả lời thắng/thua $(y_{\text{win}}, y_{\text{lose}})$:
     $$\mathcal{L}_{\text{DPO}} = -\ln \sigma\left(\beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$
   - **Lợi thế:** Loại bỏ hoàn toàn Reward Model và Critic, chỉ giữ thêm **1 bản $\pi_{\text{ref}}$ đóng băng** làm dây neo **KL-Divergence** để mô hình không nói chuyện kỳ quặc chỉ để "lách luật" điểm cao.
3. **GRPO (Group Relative Policy Optimization — nền tảng DeepSeek-R1):**
   - Với mỗi câu hỏi $x$, Policy tự sinh một nhóm **$G$ câu trả lời nháp** $(o_1, o_2, \dots, o_G)$ (ví dụ $G = 4$), chấm điểm thưởng phần thưởng cuối $(r_1 \dots r_G)$ (tính đúng toán/code/format), rồi lấy chính **trung bình và độ lệch chuẩn của cả nhóm** làm mốc so sánh (Advantage):
     $$A_i = \frac{r_i - \text{mean}(r_1 \dots r_G)}{\text{std}(r_1 \dots r_G) + \varepsilon}$$
   - **Lợi thế:** **Loại bỏ hoàn toàn mô hình Critic khổng lồ** của PPO (giảm gần một nửa VRAM bộ nhớ mô hình huấn luyện RL), đồng thời ở pha tự sinh $G$ nhánh nháp (**Rollout**), cơ chế **GQA `8Q:2KV`** phát huy tối đa tác dụng giảm 75% KV Cache cho cả 4 nhánh song song!

---

## Câu hỏi gợi ý để hỏi sâu thêm trên NotebookLM

1. *Trong LoRA, vì sao dung lượng VRAM giảm hơn một nửa (nhờ loại bỏ trạng thái AdamW 12B/param cho $W_0$) nhưng tốc độ bước train tính bằng tok/s chỉ tăng nhẹ khoảng 12–15%?*
2. *Trong Full SFT với `ignore_index = -100` trên phần câu hỏi của User, hãy giải thích vì sao token User không tạo Cross-Entropy Loss nhưng vẫn có đạo hàm đi qua lớp Attention của nó?*
3. *So sánh kiến trúc bộ nhớ và cơ chế baseline giữa PPO, DPO và GRPO: tại sao GRPO lại đặc biệt phù hợp cho các mô hình suy luận như DeepSeek-R1?*
4. *Dựa vào Chinchilla Scaling Laws ($D \approx 20N$), hãy giải thích vì sao một thí nghiệm trên bộ dữ liệu nhỏ không thể dùng để chứng minh MoE vượt trội Dense về Loss.*
