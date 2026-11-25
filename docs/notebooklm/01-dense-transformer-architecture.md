# Kiến trúc Dense Transformer Hiện đại — Từ Token thô đến Mô hình LLaMA / MiniMind

> Tài liệu này giải thích sâu toàn bộ kiến trúc **Dense Transformer** chuẩn hiện đại (dòng GPT → LLaMA → MiniMind2-Small), tập trung vào *lý do tồn tại*, *nút thắt toán học/phần cứng* và *đánh đổi kỹ thuật* của từng khối bên trong một tầng (Layer).

---

## 1. Bức tranh tổng thể: Từ câu chữ thành vector xác suất

Một Dense Transformer thuần giải mã (Decoder-only Autoregressive Transformer) xử lý chuỗi đầu vào $x = (w_0, w_1, \dots, w_{L-1})$ qua 4 chặng lớn:

1. **Tokenizer (BPE):** Tách chuỗi ký tự thành danh sách ID từ điển (`token_ids` có độ dài $L$).
2. **Token Embedding (`E ∈ ℝ^{V × d}`):** Tra bảng biến mỗi ID thành vector ẩn $h^{(0)}_t \in \mathbb{R}^d$.
3. **$N$ Tầng Transformer giống nhau (`L0 → L_{N-1}`):** Mỗi tầng gồm đúng 2 khối chính nằm trên trục **Residual Stream**:
   - **Self-Attention (có RoPE & GQA):** Trộn thông tin *giữa các vị trí* (token hiện tại đọc các token đứng trước).
   - **Feed-Forward Network (SwiGLU FFN):** Biến đổi *độc lập từng vị trí* (xử lý và truy xuất tri thức lưu trong trọng số FFN).
4. **Final RMSNorm + LM Head (`W_out ∈ ℝ^{V × d}`) + Softmax:** Chiếu vector sau tầng cuối ra $V$ điểm số (**Logits**), chuẩn hóa thành phân phối xác suất dự đoán từ tiếp theo $P(w_{t+1} \mid w_{\le t})$.

**Thông số đối chiếu thực tế:**
- **MiniMind2-Small (Dense):** `hidden_size (d) = 512`, `num_layers = 8`, `vocab_size (V) = 6,400`, `GQA = 8Q : 2KV` → **25.8M tham số** (mỗi token đi qua tính đủ **100%** = `25.8M` tham số).
- **LLaMA-3 8B:** `d = 4096`, `layers = 32`, `V = 128,256`, `GQA = 32Q : 8KV`.

---

## 2. Tokenization & Embedding: Đánh đổi của kích thước từ điển (`vocab_size`)

- **BPE (Byte-Pair Encoding):** Bắt đầu từ byte/ký tự đơn lẻ và gộp dần các cặp xuất hiện nhiều nhất thành *subword*. Từ phổ biến là 1 token, từ hiếm tách thành nhiều mảnh.
- **Đánh đổi khi chọn `vocab_size (V)`:**
  - `V` nhỏ (ví dụ `6,400` ở MiniMind): Ma trận `Embedding` và `LM Head` rất gọn (`6400 × 512 ≈ 3.28M` tham số nếu không tie weights), giúp train cực nhanh trên GPU phổ thông, nhưng một câu tiếng Việt dài tốn nhiều token hơn.
  - `V` lớn (`128,256` ở LLaMA-3): Nén câu ngắn lại (giảm chiều dài chuỗi $L$, tiết kiệm công Attention), nhưng bảng Embedding + LM Head phình to tới hàng trăm triệu tham số — điều rất nhạy cảm ở các mô hình siêu nhỏ.
- **Weight Tying (Chia sẻ trọng số Embedding và LM Head):** Nhiều LLM nhỏ dùng chung ma trận $W_{\text{out}} = E^\top$ để cắt đôi bộ nhớ lớp vào/ra và giúp không gian biểu diễn đầu vào–đầu ra nhất quán hơn.

---

## 3. Residual Stream — Mô hình tư duy quan trọng nhất thường bị bỏ qua

Nhiều tài liệu vẽ Transformer như các hộp xếp chồng nối tiếp, dễ tạo cảm giác tầng dưới xử lý xong thì "vứt" dữ liệu cũ sang tầng trên. Cách hiểu đúng hơn (từ nghiên cứu *Transformer Circuits*) là xem **Residual Stream** $x$ như một **xa lộ thông tin chung** có chiều $d$:

$$x_{\text{mid}} = x + \text{Attention}(\text{RMSNorm}(x))$$
$$x_{\text{out}} = x_{\text{mid}} + \text{FFN}(\text{RMSNorm}(x_{\text{mid}}))$$

**Ý nghĩa sâu sắc của Residual Stream:**
1. **Giao tiếp xuyên tầng (Additive Bus):** Tầng `L0` có thể viết một đặc trưng vào một "kênh" (subspace) của vector $x$, tầng `L5` đọc trực tiếp mà tầng `L1..L4` không làm mất nó đi.
2. **Đường cao tốc cho Gradient:** Khi Backward từ đầu ra về `L0`, nhờ phép cộng $+ x$, đạo hàm $\frac{\partial \mathcal{L}}{\partial x}$ đi thẳng qua các tầng với hệ số 1 mà không bị triệt tiêu (*vanishing gradient*), ngay cả ở mạng sâu hàng trăm tầng.
3. **Vai trò rẽ đôi trong mỗi tầng:** Attention là cơ chế **di chuyển thông tin theo chiều ngang (giữa các token)**; FFN là cơ chế **tính toán và tra cứu tri thức theo chiều dọc (tại từng token)** — đó là lý do MoE chỉ thay khối **FFN**, còn Attention giữ nguyên.

---

## 4. Pre-RMSNorm: Vì sao thay thế Post-LayerNorm?

- **Post-Norm (Transformer 2017 gốc):** $\text{LN}(x + f(x))$ chuẩn hóa *trên đường chính* của Residual Stream, làm gián đoạn đường truyền gradient thẳng ở pha khởi tạo → phải warmup Learning Rate rất kỹ nếu không sẽ sập hội tụ.
- **Pre-Norm (LLaMA / MiniMind):** $x + f(\text{Norm}(x))$ chỉ chuẩn hóa *nhánh rẽ* trước khi vào Attention hoặc FFN, giữ nguyên đường cao tốc Residual Stream sạch bóng.
- **RMSNorm vs LayerNorm:**
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}} \odot \gamma$$
  RMSNorm bỏ bước trừ trung bình (mean-centering) và bỏ bias $\beta$, chỉ giữ chia theo độ lớn RMS và nhân vector scale $\gamma$. Thực nghiệm cho thấy chất lượng tương đương LayerNorm nhưng giảm đáng kể chi phí đọc/ghi bộ nhớ GPU (Memory I/O).

---

## 5. Causal Self-Attention, RoPE, GQA & FlashAttention

### 5.1. Cơ chế Q, K, V và Causal Mask
Với mỗi token đầu vào đã chuẩn hóa $\tilde{x}_t$:
- Tạo $Q_t = \tilde{x}_t W_Q$ (*"Tôi đang cần tìm thông tin gì?"*), $K_t = \tilde{x}_t W_K$ (*"Tôi chứa thông tin gì?"*), $V_t = \tilde{x}_t W_V$ (*"Nội dung tôi sẽ gửi nếu được chọn"*).
- Điểm chú ý giữa token $m$ và token $n$: $\frac{Q_m \cdot K_n^\top}{\sqrt{d_k}}$ + **Causal Mask** đặt $-\infty$ cho mọi $n > m$ (token ở tương lai không được phép lộ thông tin cho token hiện tại khi train song song).

### 5.2. RoPE (Rotary Position Embedding) — Mã hóa vị trí tương đối bằng phép xoay
- **Vấn đề cũ:** Cộng vector vị trí cố định vào Embedding đầu vào (`Absolute Positional Encoding`, GPT-2) khó ngoại suy khi câu dài hơn lúc train và không thể hiện trực tiếp khoảng cách tương đối $(m - n)$.
- **Ý tưởng RoPE:** Chia vector $Q_m, K_n \in \mathbb{R}^{d_k}$ thành từng cặp 2 chiều, rồi xoay mỗi cặp của $Q_m$ đi một góc $m \cdot \theta_j$ và của $K_n$ đi góc $n \cdot \theta_j$ (các cặp tần số cao $\theta_j$ lớn nhạy với từ kế bên, tần số thấp nhạy với ngữ cảnh xa).
- **Tính chất vàng:** Khi lấy tích vô hướng trong Attention:
  $$\langle R_{\Theta, m} Q_m,\; R_{\Theta, n} K_n \rangle = Q_m^\top R_{\Theta, n - m} K_n$$
  Kết quả **chỉ phụ thuộc vào khoảng cách tương đối $(m - n)$**, bất biến với độ dài tuyệt đối, và góc xoay $m \cdot \theta_j$ hoàn toàn chỉ phụ thuộc chỉ số vị trí $m$ (không đổi theo bước train).
- **Mở rộng ngữ cảnh dài:** Các kỹ thuật như **NTK-aware scaling** và **YaRN** điều chỉnh lại phổ tần số $\theta_j$ để kéo dài context window (ví dụ từ 4K lên 128K) với rất ít bước fine-tune thêm.

### 5.3. GQA (Grouped-Query Attention) — Nút thắt thực sự nằm ở đâu?
- **MHA (Multi-Head Attention):** $H$ đầu Query đi với $H$ bộ Key và $H$ bộ Value (`8Q : 8KV`). Chất lượng cao nhất nhưng bộ nhớ lưu KV Cache lớn gấp $H$ lần.
- **MQA (Multi-Query Attention):** Tất cả $H$ đầu Query dùng chung **1** bộ K/V (`8Q : 1KV`). Nhẹ nhất nhưng dễ giảm chất lượng biểu diễn.
- **GQA (Grouped-Query Attention):** Chia nhóm trung hòa, ví dụ MiniMind2 dùng **`8Q : 2KV`** (mỗi nhóm 4 đầu Query dùng chung 1 đầu Key/Value).
- **Điểm cực kỳ quan trọng:** Khi **Pretrain thông thường**, tổng số FLOPs và VRAM activation không giảm quá mạnh vì phần lớn ma trận vẫn là phép nhân lớn cho $Q$; lợi ích sống còn của GQA giảm **75% KV Cache** (`2KV / 8KV = 25%`) bùng nổ ở **Inference (Decode)** và **GRPO Rollout**, nơi bộ nhớ K/V tỷ lệ thuận với độ dài chuỗi và batch size.

### 5.4. FlashAttention — Tối ưu IO-Aware trên GPU
- Attention tiêu chuẩn tạo ma trận $S = QK^\top \in \mathbb{R}^{L \times L}$ rất lớn và ghi/đọc qua lại bộ nhớ **HBM** (VRAM chính của GPU, băng thông chậm hơn nhiều so với SRAM trong nhân GPU).
- **FlashAttention** không đổi phép toán toán học, mà chia $Q, K, V$ thành các khối nhỏ (**tiling**) vừa đủ nằm gọn trong **SRAM** siêu nhanh trên chip, dùng kỹ thuật *online softmax* để tích lũy kết quả và ghi thẳng đầu ra cuối xuống HBM — giảm bộ nhớ từ $O(L^2)$ xuống $O(L)$ và tăng tốc thực tế 2–4× mà **không làm xấp xỉ sai số**.

---

## 6. SwiGLU FFN — Kho lưu trữ tri thức phi tuyến của Dense Transformer

Transformer cổ điển dùng FFN 2 ma trận với ReLU/GELU: $\text{FFN}(x) = W_2 \sigma(W_1 x)$ có chiều ẩn $d_{\text{ff}} = 4d$.

Các mô hình hiện đại (LLaMA, MiniMind) chuyển sang **SwiGLU** với **3 ma trận** ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$):
$$\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big) W_{\text{down}}, \quad \text{với } \text{SiLU}(z) = z \cdot \sigma(z)$$

- **Cơ chế cổng nhân (Gating):** Nhánh `gate` đóng vai trò van điều chỉnh tỷ lệ đi qua của nhánh nội dung `up`, tạo biểu diễn giàu tính chọn lọc hơn.
- **Quy tắc $\frac{8}{3}d$:** Vì có 3 ma trận thay vì 2, để giữ nguyên tổng số tham số của khối FFN ($3 \cdot d \cdot d_{\text{ff}} = 2 \cdot d \cdot 4d$), chiều ẩn thường chọn $d_{\text{ff}} \approx \frac{8}{3} d$ (làm tròn thành bội số đẹp phần cứng như `64` hoặc `128`).
- **Hạn chế dẫn tới MoE:** Ở mạng **Dense**, toàn bộ ma trận FFN ($60\text{–}70\%$ tổng tham số của mô hình lớn) phải nhân cho **mọi token** kể cả các từ nối đơn giản như `"là"`, `"của"`, `"và"`. Muốn tăng kiến thức (phình $d_{\text{ff}}$ hoặc thêm tầng), chi phí tính toán tăng tuyến tính 1:1 theo mỗi token.

---

## Câu hỏi gợi ý để hỏi sâu thêm trên NotebookLM

1. *Hãy giải thích cách nhìn Transformer như một "Residual Stream" giúp hiểu vì sao Attention và FFN đảm nhận hai vai trò khác nhau, và vì sao MoE chỉ thay FFN mà giữ nguyên Attention?*
2. *Tại sao trong công thức RoPE, tích vô hướng $Q_m \cdot K_n$ lại chỉ phụ thuộc vào khoảng cách tương đối $m - n$? Điều gì xảy ra khi gặp chuỗi dài hơn độ dài lúc pretrain?*
3. *Phân biệt sự khác nhau giữa tối ưu FLOPs (bằng số phép nhân) và tối ưu Memory I/O (FlashAttention trên HBM vs SRAM, GQA trên KV Cache).*
4. *Vì sao SwiGLU dùng $d_{\text{ff}} \approx \frac{8}{3} d$ thay vì $4d$, và chi phí bộ nhớ của nó khác gì FFN 2 ma trận?*
