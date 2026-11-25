# MoE (Mixture of Experts) — Từ Toán học Cốt lõi đến Mixtral & DeepSeek-V3

> Tài liệu này giải phẫu kiến trúc **Sparse Mixture-of-Experts (MoE)**: tại sao nó tách được "dung lượng tri thức" khỏi "chi phí tính mỗi từ", cách một chuyên gia thực sự hình thành từ Step 0, các thuật toán cân bằng tải từ Switch Transformer tới DeepSeek-V3, và những nút thắt phần cứng thực tế.

---

## 1. Động lực cốt lõi: Tách rời "Dung lượng" và "Chi phí tính" (Conditional Computation)

Trong một mô hình **Dense**, hàm ánh xạ luôn kích hoạt toàn bộ tham số $N_{\text{total}}$ cho mọi token đi qua:
$$\text{FLOPs/token} \propto 2 \cdot N_{\text{total}}$$

Điều này tạo ra một **trần đánh đổi cứng**: muốn mô hình nhớ nhiều tri thức đa lĩnh vực hơn (tăng $N_{\text{total}}$), ta bắt buộc phải trả chi phí tính toán lớn hơn cho từng token một.

**Sparse MoE** phá vỡ liên kết 1:1 này bằng cách thay khối FFN khổng lồ duy nhất bằng:
- **$N$ Routed Experts** ($E_0, E_1, \dots, E_{N-1}$): các mạng FFN nhỏ song song nhau.
- **1 Router (Gate Network):** mạng tuyến tính nhỏ quyết định với *mỗi token cụ thể* chỉ chọn đúng **$K \ll N$** chuyên gia tốt nhất.
- *(Tùy chọn)* **$N_s$ Shared Experts:** các khối FFN nhỏ luôn bật cho mọi token (có ở DeepSeekMoE, Qwen-MoE, MiniMind2-MoE; **không** có ở Mixtral 8x7B hay MiniMind-3).

**Kết quả định lượng:**
- **MiniMind2-MoE:** `4 Routed (Top-2) + 1 Shared`, `hidden = 640`, `8 layers` → **`145.0M` tổng tham số**, nhưng mỗi token chỉ tính **`45.2M` tham số kích hoạt** (~31%). So với một mạng Dense giả định cùng cỡ `145M`, MoE giảm ~**`3.2×`** lượng tính toán mỗi token.
- **Mixtral 8x7B:** `8 experts / tầng`, `Top-2`, không Shared → `46.7B` tổng tham số, chỉ kích hoạt `12.9B` mỗi token.
- **DeepSeek-V3:** `256` routed experts mịn (`Top-8`) + `1` shared expert → **`671B`** tổng tham số, chỉ kích hoạt **`37B`** mỗi token (~5.5%).

---

## 2. Phương trình chuẩn của một tầng MoE

Với vector đầu vào sau Attention + RMSNorm tại một token $x \in \mathbb{R}^d$:

1. **Router tính điểm chọn cho $N$ chuyên gia:**
   $$p = \text{Softmax}\left(\frac{W_g x + b_g}{T}\right) \in \mathbb{R}^N, \quad W_g \in \mathbb{R}^{N \times d}$$
2. **Chọn Top-K và chuẩn hóa lại trọng số (`norm_topk_prob = True`):**
   Gọi $\mathcal{T}_K(p)$ là tập chỉ số $K$ phần tử lớn nhất của $p$. Trọng số đầu ra của các chuyên gia được chọn là:
   $$w_i = \frac{p_i}{\sum_{j \in \mathcal{T}_K(p)} p_j}, \quad \forall i \in \mathcal{T}_K(p)$$
3. **Tổng hợp kết quả tại $\Sigma$:**
   $$y = \sum_{i \in \mathcal{T}_K(p)} w_i \cdot E_i(x) \;+\; \text{Shared}(x)$$

**Lưu ý quan trọng dễ hiểu nhầm:**
- Bản sao vector $x$ gửi **vào** các chuyên gia được chọn là **giống hệt nhau**. Chỉ có **kết quả đi ra** $E_i(x)$ mới được nhân trọng số $w_i$ trước khi cộng ở $\Sigma$ (giữ độ lớn gradient ổn định qua nhánh router).
- **Mỗi tầng có Router độc lập riêng** ($W_g^{(0)} \dots W_g^{(L-1)}$), nên cùng một từ `"của"` có thể đi vào `(E1, E2)` ở `L0` nhưng sang `(E0, E3)` ở `L4`.

---

## 3. Vòng đời của Chuyên gia: Số lượng cố định từ Step 0 → Chuyên môn hóa tự nổi lên

### 3.1. Số chuyên gia có phải định nghĩa sẵn từ đầu không?
**Có — cố định 100% trước Step 0.** Số lượng expert $N$ (`4` ở MiniMind2, `8` ở Mixtral, `256` ở DeepSeek-V3) và số lượng kích hoạt $K$ là **siêu tham số kiến trúc** giống như số tầng hay `hidden_size`. Trong suốt quá trình train, mạng **không tự sinh thêm hay xóa bớt** bất kỳ khối chuyên gia nào.

### 3.2. Khi nào một "chuyên gia" thực sự được hình thành?
Cần phân biệt rạch ròi **cái vỏ (ma trận trọng số)** và **cái hồn (chuyên môn hóa)**:

1. **Step 0 (Khởi tạo ngẫu nhiên):** Toàn bộ $N$ vỏ FFN đã được cấp phát trên GPU, nhưng chỉ chứa nhiễu ngẫu nhiên; Router chia tải gần như đều (~25/25/25/25 ở `N=4`) một cách vô nghĩa. Lúc này chưa hề có "chuyên gia" theo nghĩa chức năng.
2. **Phá vỡ đối xứng (Symmetry Breaking — Pretrain sớm):**
   Với một mẫu ngữ cảnh $x$, do sai lệch nhỏ ở khởi tạo, giả sử $E_1(x)$ tình cờ giúp giảm Cross-Entropy Loss tốt hơn $E_2(x)$ một chút. Theo quy tắc chuỗi qua trọng số $w_i$, đạo hàm $\frac{\partial \mathcal{L}_{\text{CE}}}{\partial p_1} < \frac{\partial \mathcal{L}_{\text{CE}}}{\partial p_2}$ làm Router tăng điểm $p_1$ cho các mẫu gần $x$.
3. **Vòng lặp tự củng cố (*Rich-Get-Richer*):**
   Khi $E_1$ nhận nhiều mẫu giống $x$ hơn, nó nhận nhiều bước cập nhật gradient trên cụm mẫu đó hơn → càng dự đoán chính xác hơn trên cụm mẫu đó → Router càng tin tưởng giao thêm cho $E_1$. Nhờ vậy chuyên môn hóa **tự nổi lên (emergent)** từ dữ liệu chảy qua mà không cần con người gán nhãn.
4. **Đóng băng ở LoRA & Inference:** Trong LoRA (thường chỉ bọc `q_proj`, `o_proj`), cả Router lẫn các Expert đều bị đóng băng; ở Inference toàn mạng đứng yên.

### 3.3. Router thực tế học chia theo "chủ đề con người" hay theo cái gì?
- Các nghiên cứu thực nghiệm trên Mixtral và Switch Transformer cho thấy: ở các tầng nông (`L0–L2`), Router thường nhóm theo **đặc điểm cú pháp, dấu câu, khoảng trắng, thụt lề code hoặc vị trí hình thái từ** chứ không chia gọn gàng kiểu "E0 = Địa lý, E1 = Ẩm thực". Ở các bộ dữ liệu thử nghiệm nhỏ tách biệt từ vựng theo chủ đề, màu chủ đề trùng khớp nhiều hơn chỉ vì từ vựng các miền không giao nhau.

---

## 4. Vấn đề Sụp đổ Tải (Load Collapse) & Các cơ chế Cân bằng Tải

Nếu để Router chỉ tối ưu theo $\mathcal{L}_{\text{CE}}$ thuần túy, vòng lặp tự củng cố sẽ khiến 1–2 chuyên gia gom hết mọi token trong khi các chuyên gia còn lại gần như **chết đứng (never updated)**.
- Với Top-1: 1 expert có thể ôm **100%** lượt.
- Với Top-2 (mỗi token phải chọn 2 expert khác nhau): mỗi expert nhận tối đa **50%** tổng lượt chọn → trạng thái sụp đổ điển hình là **`50% / 50% / 0% / 0%`**.
- **Thước đo Entropy hiệu dụng:**
  $$N_{\text{eff}} = \exp\Big(-\sum_{i=0}^{N-1} \text{load}_i \ln \text{load}_i\Big)$$
  Khi chia đều: $N_{\text{eff}} = 4.0$; khi sụp đổ `50/50/0/0`: $N_{\text{eff}} = 2.0$. Lúc này GPU vẫn gánh VRAM cho đủ 4 expert nhưng dung lượng học thực tế bị cắt đôi.

### Tiến hóa 4 kỹ thuật cân bằng tải trong MoE hiện đại

1. **Auxiliary Load Balancing Loss (`seq_aux` trong Switch Transformer & MiniMind2):**
   $$\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{i=0}^{N-1} ce_i \cdot P_i, \quad \text{với } ce_i = \frac{\text{count}_i}{T \cdot K / N}, \quad P_i = \frac{1}{T}\sum_{t=1}^T p_{t,i}$$
   Trong đó $\text{count}_i$ là số token chọn expert $i$ (không khả vi), còn $P_i$ là trung bình xác suất softmax (khả vi). Nhân $ce_i \cdot P_i$ buộc gradient đẩy giảm $P_i$ của những expert đang bị quá tải $ce_i > 1$.
   - Với `Top-2`, giá trị $\sum ce_i P_i$ đạt `1.0` khi chia đều hoàn toàn ($\mathcal{L}_{\text{aux}} \approx \alpha$) và tiến tới `2.0` khi sụp đổ `50/50/0/0` ($\mathcal{L}_{\text{aux}} \approx 2\alpha$).
   - **Nhược điểm:** $\alpha$ quá lớn sẽ **nhiễu gradient học ngôn ngữ** (ép từ sai ngữ cảnh vào expert rảnh chỉ để đủ KPI); $\alpha$ quá nhỏ lại không chống nổi collapse.

2. **Router Z-Loss (ST-MoE):**
   $$\mathcal{L}_z = \frac{1}{T}\sum_{t=1}^T \Big(\ln \sum_{i=1}^N e^{z_{t,i}}\Big)^2$$
   Phạt khi các logit đầu vào softmax của router phình quá lớn, giúp ngăn tràn số trên `bf16` và giữ quyết định chọn expert ổn định khi train ở quy mô lớn.

3. **Capacity Factor & Token Dropping (Cách cũ theo kiểu GShard/Switch):**
   Giới hạn mỗi expert trong 1 batch chỉ được xử lý tối đa $\text{Capacity} = \text{CF} \times \frac{T \cdot K}{N}$ token (ví dụ `CF = 1.25`). Nếu bị tràn, các token dư sẽ **bị bỏ qua (dropped)** ở tầng đó (chỉ đi qua residual stream) → làm mất thông tin token quan trọng. Các hệ thống mới hiện tìm cách giảm hoặc loại bỏ token dropping (`CF` lớn hoặc dropless Megablocks).

4. **Auxiliary-Loss-Free Load Balancing bằng Dynamic Bias (DeepSeek-V3):**
   Thay vì cộng $\mathcal{L}_{\text{aux}}$ vào hàm mất mát (làm méo gradient của Router), DeepSeek-V3 cộng một vector **bias động $b^{\text{bal}}_i$** *chỉ lúc chọn Top-K* (không dùng để nhân trọng số đầu ra $w_i$):
   $$\text{chọn } \mathcal{T}_K(p + b^{\text{bal}}), \quad \text{nhưng } y = \sum_{i \in \mathcal{T}_K} w_i E_i(x)$$
   Sau mỗi bước train, nếu expert $i$ đang quá tải thì hạ nhẹ $b^{\text{bal}}_i$ đi một lượng $\gamma$; nếu quá rảnh thì tăng nhẹ $b^{\text{bal}}_i$. Kết quả: tải vẫn cân bằng hoàn hảo mà **không hề làm nhiễu gradient ngôn ngữ**.

---

## 5. Xu hướng kiến trúc MoE mới: Fine-Grained Experts + Shared Experts

Tại sao **DeepSeekMoE / DeepSeek-V3** lại chia nhỏ tới **256 experts (chọn 8)** thay vì **8 experts to (chọn 2)** như Mixtral?

1. **Độ linh hoạt tổ hợp (Combinatorial Specialization):**
   - Chọn 2 trong 8 cho $\binom{8}{2} = 28$ tổ hợp tri thức khác nhau mỗi token.
   - Chia mỗi expert nhỏ lại 4 lần thành 32 experts và chọn 8 (cùng tổng FLOPs!) cho $\binom{32}{8} \approx 10.5 \text{ triệu}$ tổ hợp — các chuyên gia trở nên chuyên biệt tinh vi hơn thay vì "ôm đồm" nhiều khái niệm vào 1 khối FFN to.
2. **Tách tri thức phổ quát bằng Shared Expert:**
   Khi mọi câu đều cần ngữ pháp cơ bản, dấu câu, logic chung, nếu không có Shared Expert thì các Routed Experts buộc phải **học lặp lại cùng một kiến thức nền** trong nhiều khối khác nhau. Đưa phần chung ra `Shared Expert` giúp các `Routed Experts` rảnh tay chuyên tâm vào tri thức đặc thù.

---

## 6. Thực tế Hệ thống & Phần cứng: Mặt trái ít được nhắc đến của MoE

1. **Bùng nổ VRAM (Memory Footprint):** Dù chỉ chạy `45.2M/145M` tham số/từ (hay `37B/671B` ở DeepSeek-V3), GPU vẫn phải nạp **toàn bộ `100%` trọng số của cả $N$ chuyên gia** vào VRAM (và khi train phải cấp phát Grads + AdamW cho toàn bộ $N$ chuyên gia!). Trên một máy cá nhân ít VRAM, MoE **khó chạy hơn** Dense cùng lượng FLOPs/token.
2. **All-to-All Communication Overhead:** Khi train trên cụm nhiều GPU bằng **Expert Parallelism (EP)** (mỗi GPU giữ vài expert), mỗi token phải gửi vector ẩn qua mạng liên kết (NVLink/InfiniBand) tới đúng GPU giữ expert được chọn rồi gửi ngược về → dễ bị **nút thắt mạng (All-to-All latency)** và hiện tượng **straggler** (mọi GPU phải đợi GPU nào bị dồn nhiều token nhất chạy xong).
3. **Bất ổn khi Batch Size nhỏ:** Lúc **Decode 1 người dùng** (`batch_size = 1`), một token chỉ kích hoạt $K$ expert, nên băng thông GPU vẫn chỉ đọc lượng trọng số ít (nhanh!), *nhưng* khi phục vụ đồng thời nhiều người dùng ở nhiều chủ đề khác nhau (`batch_size = 64`), gần như toàn bộ $N$ expert đều bị gọi trong 1 bước, làm giảm độ tập trung cache và đòi hỏi kernel fused MoE chuyên biệt (Grouped GEMM).

---

## Câu hỏi gợi ý để hỏi sâu thêm trên NotebookLM

1. *Phân biệt "cái vỏ" và "cái hồn" của chuyên gia MoE từ Step 0 tới khi hội tụ: nếu không có nhãn chủ đề từ dữ liệu, gradient nào cụ thể đã khiến Router và các Expert tự phân hóa?*
2. *So sánh toán học và hiệu ứng phụ giữa Aux Loss (`seq_aux`), Capacity Factor (Token Dropping) và phương pháp Auxiliary-Loss-Free Bias của DeepSeek-V3.*
3. *Vì sao DeepSeekMoE chia thành nhiều Fine-grained Experts kết hợp Shared Expert lại tối ưu tham số tốt hơn thiết kế Coarse Experts (`8` chọn `2`) của Mixtral?*
4. *Vì sao nói MoE rất có lợi khi phục vụ `batch_size = 1` (chỉ đọc $K$ expert từ VRAM) hoặc train cụm lớn trên dữ liệu khổng lồ, nhưng lại bất lợi về VRAM trên một laptop GPU 4GB–8GB?*
