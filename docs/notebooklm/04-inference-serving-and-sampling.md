# Cách AI Trả về Kết quả (Inference) — Prefill, Decode, KV Cache, MLA & Sampling

> Đây là mảnh ghép thường bị thiếu khi học về LLM. Khi trả lời người dùng (**Inference**), toàn bộ trọng số đã đóng băng: **không có Cross-Entropy Loss, không có Backpropagation, không có AdamW**. Thay vào đó, mô hình chạy theo một vòng lặp tự hồi quy (**Autoregressive Loop**) gồm 2 pha có nút thắt phần cứng trái ngược hoàn toàn.

---

## 1. Khác biệt sống còn giữa Training và Inference

| Thuộc tính | 🏋️ Huấn luyện (Training) | 💬 Trả về kết quả (Inference) |
| :--- | :--- | :--- |
| **Cách xử lý câu** | Biết trước đáp án → đọc **tất cả $L$ token song song** 1 lần. | Chưa biết câu trả lời → sinh **từng token một** ($w_t$ xong mới tính được $w_{t+1}$). |
| **Chiều truyền** | `Forward → Loss → Backward → AdamW`. | Chỉ có **Forward** (`Prefill → Decode`). |
| **Trọng số mạng** | Thay đổi sau mỗi bước. | **Đóng băng 100%.** |
| **Bộ nhớ tăng thêm** | Gradients (`2B`) + AdamW (`12B`) + Activations. | **KV Cache** (lớn dần theo từng token của cuộc hội thoại). |

---

## 2. Hai pha của một lần Trả lời: Prefill vs. Decode

Giả sử bạn hỏi: *"Thủ đô của Việt Nam là gì?"* (`6` token prompt), và AI trả lời *"Hà Nội"* (`<eos>`).

### Pha 1 — Prefill (Đọc hiểu câu hỏi ban đầu) — *Compute-Bound*
- Mô hình nhận toàn bộ `6` token prompt cùng lúc (`[1, 6, d]`).
- Giống Forward khi train, cả 6 token được nhân ma trận **song song**: tính $Q, K, V$ cho cả 6 token trong 1 lượt qua `L0 → L_{N-1}`.
- **Đầu ra của Prefill:**
  1. Nạp sẵn ma trận $K_{\le 5}, V_{\le 5}$ của cả 6 token vào **KV Cache** ở từng tầng.
  2. Lấy vector tại vị trí cuối cùng (`vị trí 5`) chiếu qua LM Head để sinh **token trả lời đầu tiên** (`"Hà"`).
- **Đặc tính GPU:** Phép nhân ma trận-ma trận (**GEMM**), tái sử dụng trọng số nhiều lần trên cả batch token → bị giới hạn bởi **tốc độ tính toán (Compute-bound)**. Thời gian hoàn tất pha này gọi là **TTFT (*Time To First Token*)**.

### Pha 2 — Decode (Sinh từng chữ kế tiếp) — *Memory-Bandwidth-Bound*
- Từ bước thứ 2 trở đi, ở mỗi bước $t$ ta chỉ đưa **duy nhất `1` token mới nhất** (`"Hà"`) vào mạng (`[1, 1, d]`)!
- Tại mỗi tầng:
  1. Kim **RoPE** xoay góc $Q_t, K_t$ theo vị trí hiện tại `m = t`.
  2. Nối $K_t, V_t$ mới vào cuối **KV Cache** (`[K_0..K_t]`, `[V_0..V_t]`), rồi cho $Q_t$ truy vấn trực tiếp toàn bộ KV Cache **mà không bao giờ tính lại $K, V$ của các token cũ**.
  3. *(Nếu là MoE)*: Router chỉ kích hoạt $K$ chuyên gia (`45.2M / 145M` ở MiniMind2) cho riêng token này.
- Vòng lặp lặp lại tới khi chọn trúng `<eos>` hoặc chạm `max_new_tokens`.
- **Đặc tính GPU (cực kỳ quan trọng):** Mỗi bước chỉ xử lý `1` token (phép nhân vector-ma trận **GEMV**), nhưng GPU phải **lôi toàn bộ trọng số mô hình + toàn bộ KV Cache từ VRAM (HBM) lên nhân xử lý chỉ để tính cho 1 từ**! Số phép tính rất ít so với lượng byte phải đọc → bị nghẽn bởi **băng thông bộ nhớ VRAM (*Memory-Bandwidth-Bound*)**. Thời gian mỗi bước này là **TPOT (*Time Per Output Token*)**.

---

## 3. KV Cache: Công thức chính xác & Tiến hóa `MHA → MQA → GQA → MLA`

### 3.1. Công thức tính dung lượng KV Cache
Với mỗi token nằm trong cửa sổ ngữ cảnh, ở mỗi tầng ta lưu 2 tensor `K` và `V`:
$$\text{KV Cache (bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_elem}$$

**Ví dụ cụ thể (precision `bf16 = 2 bytes`, `seq_len = 8,192`, `batch = 1`):**
- Một mô hình cỡ 8B (`32` layers, `d_head = 128`):
  - Nếu dùng **MHA** (`32` KV heads): $2 \times 32 \times 32 \times 128 \times 8192 \times 2 \approx 4.29 \text{ GB}$ **chỉ cho KV Cache của 1 người dùng**! Phục vụ `batch = 16` tốn tới `~68.7 GB` riêng KV Cache.
  - Nếu dùng **GQA 4:1** (`8` KV heads — giống tỷ lệ `8Q : 2KV` ở MiniMind2): giảm **75%** xuống còn **`~1.07 GB` / người dùng**.

### 3.2. Tiến hóa 4 kiến trúc thu gọn KV Cache
1. **MHA (`Multi-Head Attention`, GPT-3):** $n_{\text{kv}} = n_q$. Chất lượng tốt, KV Cache khổng lồ.
2. **MQA (`Multi-Query Attention`, PaLM / Falcon):** $n_{\text{kv}} = 1$. Giảm KV Cache tối đa, nhưng dễ tụt chất lượng nếu nén quá gắt.
3. **GQA (`Grouped-Query Attention`, LLaMA-2/3, MiniMind2 `8Q:2KV`):** Chia $n_q$ thành các nhóm dùng chung 1 đầu KV. Cân bằng gần như nguyên vẹn chất lượng MHA nhưng cắt 75–87.5% KV Cache.
4. **MLA (`Multi-Head Latent Attention`, DeepSeek-V2 / V3):** Nén cả vector Key và Value chung vào một vector tiềm ẩn (**compressed latent $c_{KV}$**) có số chiều rất nhỏ (ví dụ `512`) rồi mới phục hồi ngược bằng phép chiếu hấp thụ vào ma trận Query khi nhân (*weight absorption*) → giảm **~93% KV Cache** so với MHA mà chất lượng còn vượt cả MHA tiêu chuẩn!

### 3.3. PagedAttention (vLLM) — Giải quyết phân mảnh bộ nhớ serving
Trước PagedAttention, server phải cấp phát một đoạn VRAM liên tiếp dài bằng `max_seq_len` cho mỗi request → **lãng phí 60–80% VRAM KV Cache** vì phần lớn câu trả lời ngắn hơn mức tối đa rất nhiều.
**PagedAttention** học theo bộ nhớ ảo (Virtual Memory) của hệ điều hành: cắt KV Cache thành từng **trang cố định kích thước (ví dụ 16 tokens/block)** nằm rải rác trong VRAM và ánh xạ bằng **Block Table**, giúp gần như không lãng phí bộ nhớ và dễ dàng **chia sẻ tiền tố hệ thống (Prefix Sharing / Copy-on-Write)** giữa nhiều người dùng.

---

## 4. Từ Logits thành Chữ: Chiến lược Chọn từ (Decoding & Sampling)

Tại đầu ra của bước Decode, mạng trả về vector **Logits** $z \in \mathbb{R}^V$. Cách chuyển $z$ thành chữ cụ thể quyết định độ chính xác hay sáng tạo của câu trả lời:

1. **Temperature ($T$):** Chia logits trước khi vào Softmax:
   $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - **$T \to 0$ (Greedy Decoding):** Luôn chọn đúng từ điểm cao nhất ($\arg\max$). Nhất quán, hợp bài toán code/toán, nhưng dễ lặp câu.
   - **$T = 0.6 \sim 0.8$:** Phân phối cân bằng tự nhiên giữa mạch lạc và đa dạng.
   - **$T > 1.0$:** Làm phẳng phân phối xác suất, các từ hiếm có cơ hội được chọn cao hơn → sáng tạo nhưng dễ "bịa" hoặc lạc đề.

2. **Top-k Sampling:** Chỉ giữ lại $k$ từ có xác suất cao nhất (ví dụ `k = 40`), đặt xác suất phần đuôi về 0 rồi chuẩn hóa lại trước khi bốc thăm.

3. **Top-p (Nucleus Sampling):** Thay vì cắt số lượng cứng $k$, chọn tập nhỏ nhất các từ đứng đầu sao cho **tổng xác suất tích lũy $\ge p$** (ví dụ `p = 0.9`). Khi mô hình rất chắc chắn (1 từ chiếm 95%), tập chỉ còn 1–2 từ; khi mô hình phân vân, tập tự mở rộng linh hoạt.

4. **Min-p Sampling:** Cắt mọi từ có xác suất nhỏ hơn $p_{\min} \times p_{\max}$ (so với từ đứng đầu). Rất hiệu quả khi kết hợp Temperature cao vì chặn đứng nhiễu rác nhưng vẫn giữ các phương án cạnh tranh tốt.

5. **Beam Search vs. Sampling:** Beam Search giữ $B$ nhánh có tổng xác suất chuỗi cao nhất song song — từng phổ biến cho dịch máy, nhưng với hội thoại mở thường tạo văn phong nhàm chán, lặp lại.

---

## 5. Các kỹ thuật Tăng tốc Thực chiến ở Inference

1. **Weight Quantization (`FP8 / INT8 / INT4 - AWQ / GPTQ`):**
   Vì pha Decode bị nghẽn bởi **băng thông đọc trọng số từ VRAM**, nén trọng số từ `16-bit` xuống `4-bit` giảm tới **4× lượng byte phải chuyển mỗi bước**, giúp tốc độ sinh token/giây (`tok/s`) trên máy cá nhân tăng gần tương ứng 2.5–3.5× (dù số FLOPs không giảm!).
2. **Speculative Decoding (Giải mã Suy đoán):**
   Dùng một mô hình **nháp siêu nhỏ (Draft Model)** sinh nhanh 4–5 token liên tiếp, sau đó đưa cả `5` token đó vào **mô hình lớn (Target Model)** kiểm tra **song song chỉ trong 1 bước Forward kiểu Prefill**! Nhờ cơ chế *rejection sampling*, đầu ra giữ **đúng 100% phân phối xác suất gốc** của mô hình lớn nhưng tốc độ sinh tăng 2–3×.
3. **Continuous Batching (Iteration-level Scheduling):**
   Thay vì đợi cả batch 16 người đều sinh tới `<eos>` mới nhận người mới, ngay khi 1 request kết thúc ở giữa bước, slot KV Cache của nó được giải phóng và chèn request mới vào ở đúng iteration tiếp theo.

---

## Câu hỏi gợi ý để hỏi sâu thêm trên NotebookLM

1. *Tại sao cùng một phép toán Attention, pha Prefill lại bị Compute-bound còn pha Decode lại bị Memory-Bandwidth-bound? Điều này giải thích thế nào cho việc lượng tử hóa trọng số xuống 4-bit (AWQ/GPTQ) giúp tăng tốc Decode?*
2. *Tính tay lượng KV Cache tiết kiệm được khi chuyển từ MHA sang GQA `8Q:2KV` và sang MLA (DeepSeek-V2/V3) ở độ dài ngữ cảnh 32K tokens.*
3. *Phân biệt Greedy Decoding, Temperature Scaling, Top-k, Top-p và Min-p: khi viết code Python cần độ chính xác tuyệt đối nên đặt các tham số này như thế nào?*
4. *Speculative Decoding làm sao tăng tốc độ sinh từ mà vẫn đảm bảo phân phối xác suất của đầu ra khớp chính xác với mô hình gốc?*
