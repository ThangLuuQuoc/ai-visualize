# Bản đồ Paper Gốc & Bộ Câu hỏi Đào sâu trên NotebookLM

> Tài liệu này tổng hợp **14 bài báo khoa học nền tảng** theo 4 mạch chủ đề (Dense, MoE, Training/Alignment, Inference/Serving), kèm **12 bộ câu hỏi mẫu** tối ưu để bạn chat phản biện, tạo Mind Map và Audio Overview trực tiếp trên NotebookLM.

---

## 1. Bản đồ 14 Paper Cốt lõi (Bài toán cũ → Đột phá → Cái giá)

### Mạch 1 — Nền tảng Dense Transformer
1. **Attention Is All You Need** *(Vaswani et al., 2017)*
   - *Bài toán cũ:* RNN/LSTM xử lý tuần tự theo thời gian nên chậm khi train song song và khó giữ ngữ cảnh xa.
   - *Đột phá:* Cơ chế Self-Attention + Multi-Head xử lý song song toàn bộ chuỗi với độ dài đường truyền $O(1)$.
   - *Cái giá:* Độ phức tạp bộ nhớ/tính toán Attention tăng bậc hai $O(L^2)$ theo độ dài chuỗi.
2. **RoFormer — Rotary Position Embedding (RoPE)** *(Su et al., 2021)*
   - *Bài toán cũ:* Positional Embedding cộng trực tiếp khó ngoại suy độ dài và không mã hóa tự nhiên khoảng cách tương đối $(m - n)$.
   - *Đột phá:* Mã hóa vị trí bằng phép xoay vector `Q, K` theo cặp góc $m \cdot \theta_j$ → tích vô hướng chỉ phụ thuộc khoảng cách tương đối $m - n$.
   - *Cái giá:* Cần điều chỉnh phổ tần số (NTK/YaRN) nếu muốn ngoại suy vượt xa độ dài pretrain.
3. **GLU Variants Improve Transformer (SwiGLU)** *(Shazeer, 2020)*
   - *Bài toán cũ:* FFN 2 ma trận với ReLU/GELU kém linh hoạt trong việc lọc thông tin theo ngữ cảnh.
   - *Đột phá:* SwiGLU dùng 3 ma trận kết hợp cổng nhân SiLU, tăng rõ rệt chất lượng trên mỗi FLOPs.
   - *Cái giá:* Cần thu hẹp chiều ẩn còn $\approx \frac{8}{3}d$ để cân bằng số tham số.
4. **GQA: Training Generalized Multi-Query Transformer** *(Ainslie et al., 2023)*
   - *Bài toán cũ:* MHA tốn KV Cache khổng lồ khi Inference; MQA lại giảm chất lượng.
   - *Đột phá:* Nhóm nhiều Query Heads chia sẻ chung một Key/Value Head (`8Q:2KV` hoặc `32Q:8KV`), có thể uptrain từ checkpoint MHA cũ.
   - *Cái giá:* Vẫn chưa nén sâu KV Cache bằng phương pháp latent compression (MLA sau này).
5. **FlashAttention (1 & 2)** *(Dao et al., 2022–2023)*
   - *Bài toán cũ:* Attention tiêu chuẩn đọc/ghi ma trận $L \times L$ qua lại VRAM (HBM) làm nghẽn I/O.
   - *Đột phá:* Chia khối (tiling) vừa SRAM trên chip + online softmax, giữ kết quả chính xác tuyệt đối nhưng giảm bộ nhớ xuống $O(L)$ và nhanh 2–4×.
   - *Cái giá:* Cần viết CUDA kernel chuyên sâu phụ thuộc kiến trúc GPU cụ thể.

### Mạch 2 — Kiến trúc & Cân bằng MoE
6. **Switch Transformers** *(Fedus et al., 2021)*
   - *Bài toán cũ:* Các mô hình MoE đời đầu chọn Top-2 và truyền tải nặng khó scale lên hàng trăm tỷ – nghìn tỷ tham số.
   - *Đột phá:* Đơn giản hóa xuống định tuyến Top-1, giới thiệu dạng chuẩn `auxiliary load balancing loss` và phân tích độ ổn định fp16/bf16.
   - *Cái giá:* Dùng Capacity Factor làm rớt token dư (token dropping) và dễ bị sụp tải nếu hệ số aux không khéo.
7. **Mixtral of Experts** *(Mistral AI, 2024)*
   - *Bài toán cũ:* Cần mô hình mã nguồn mở gọn hơn Llama-2 70B nhưng tốc độ ngang mô hình 13B.
   - *Đột phá:* `8` experts/tầng, `Top-2`, tổng `46.7B` nhưng chỉ kích hoạt `12.9B`/token; vượt Llama-2 70B ở nhiều benchmark với tốc độ suy luận nhanh hơn gấp nhiều lần.
   - *Cái giá:* Vẫn đòi hỏi gần ~`90GB+` VRAM ở fp16 (hoặc ~26GB ở 4-bit) để giữ cả 8 chuyên gia.
8. **DeepSeekMoE** *(Dai et al., 2024)*
   - *Bài toán cũ:* Coarse MoE (`8` experts to) làm một chuyên gia phải ôm nhiều tri thức lẫn lộn và lặp kiến thức chung.
   - *Đột phá:* Chia nhỏ thành **Fine-grained Experts** + tách riêng **Shared Experts** luôn bật cho kiến thức phổ quát.
   - *Cái giá:* Phức tạp hóa thao tác gom/chia token trên nhiều expert nhỏ khi viết kernel GPU.
9. **DeepSeek-V3 Technical Report** *(DeepSeek-AI, 2024)*
   - *Bài toán cũ:* Aux Loss làm nhiễu gradient ngôn ngữ; MHA/GQA vẫn tốn KV Cache ở ngữ cảnh dài.
   - *Đột phá:* `256` experts Top-8 (`37B / 671B` active), **Auxiliary-Loss-Free Balancing** bằng bias động, kết hợp **MLA** nén KV Cache ~93% và **FP8 Mixed Precision Training**.
   - *Cái giá:* Đòi hỏi hạ tầng mạng cụm lớn tối ưu All-to-All communication và multi-node pipelining tinh vi.

### Mạch 3 — Huấn luyện, Scaling & Alignment
10. **Training Compute-Optimal LLMs (Chinchilla)** *(Hoffmann et al., 2022)*
    - *Bài toán cũ:* Các mô hình như Gopher (`280B`) được train thiếu token (`300B` tokens), vừa to vừa kém hiệu quả.
    - *Đột phá:* Chứng minh `N` (tham số) và `D` (tokens) phải tăng cân xứng theo `D ≈ 20N` để tối ưu hiệu năng trên cùng ngân sách compute.
    - *Cái giá:* Để giảm chi phí inference sau này, các dòng LLaMA thường chấp nhận "over-train" vượt xa `20N` (ví dụ LLaMA-3 8B train tới `15T` tokens!).
11. **LoRA: Low-Rank Adaptation** *(Hu et al., 2021)* & **QLoRA** *(Dettmers et al., 2023)*
    - *Bài toán cũ:* Full fine-tuning đòi hỏi lưu AdamW 12B/param cho toàn bộ trọng số gốc, bất khả thi trên GPU đơn lẻ.
    - *Đột phá:* Đóng băng $W_0$ (hoặc nén $W_0$ xuống 4-bit NF4 trong QLoRA), chỉ train cặp ma trận hạng thấp $B \cdot A$ (`0.1–1%` tham số), cắt sâu VRAM mà vẫn giữ ~chất lượng full SFT.
    - *Cái giá:* Tốc độ bước train chỉ tăng nhẹ (~12–15%) vì sóng gradient kích hoạt vẫn đi qua $W_0$; đôi khi hội tụ chậm hơn ở tác vụ hoàn toàn mới.
12. **DPO: Direct Preference Optimization** *(Rafailov et al., 2023)*
    - *Bài toán cũ:* PPO cần huấn luyện riêng Reward Model và nuôi thêm Critic Network cực kỳ bất ổn.
    - *Đột phá:* Biến đổi toán học trực tiếp trên cặp `(y_win, y_lose)` giữ ràng buộc KL với $\pi_{\text{ref}}$, không cần Reward Model hay sampling vòng lặp RL phức tạp.
    - *Cái giá:* Dữ liệu cặp ưu tiên là tĩnh (offline), kém khả năng tự sửa sai khi mô hình tự sinh chuỗi suy luận dài (như toán/code).
13. **DeepSeekMath / GRPO (Group Relative Policy Optimization)** *(Shao et al., 2024)*
    - *Bài toán cũ:* Với bài toán suy luận dài (Chain-of-Thought), Critic Network trong PPO tốn tới 50% VRAM nhưng ước lượng giá trị cuối câu lại thiếu chính xác.
    - *Đột phá:* Sinh nhóm $G$ câu trả lời nháp cho cùng prompt, dùng điểm chuẩn hóa tương đối $A_i = (r_i - \mu)/\sigma$ thay cho Critic Network (trụ cột tạo nên DeepSeek-R1).
    - *Cái giá:* Cần sinh nhiều rollout ($G = 4 \sim 64$) mỗi câu hỏi, đẩy tải lớn lên tốc độ Inference & KV Cache trong khi train RL.

### Mạch 4 — Phục vụ & Tăng tốc Inference
14. **PagedAttention (vLLM)** *(Kwon et al., 2023)*
    - *Bài toán cũ:* Cấp phát tĩnh KV Cache liên tiếp làm lãng phí 60–80% VRAM và giới hạn số request đồng thời.
    - *Đột phá:* Quản lý KV Cache theo trang bộ nhớ không liên tiếp (như OS Virtual Memory) + Prefix Sharing → tăng thông lượng phục vụ 2–4×.
    - *Cái giá:* Tốn thêm một bước tra bảng chỉ mục block trong CUDA attention kernel.

---

## 2. Bộ Prompt Gợi ý Dùng Trực tiếp trên NotebookLM

Sau khi upload 6 file này vào một Notebook trên NotebookLM, bạn có thể copy-paste các prompt sau để khai thác sâu hơn:

### A. Prompt Tổng hợp & Đối chiếu (Hiểu bản chất)
1. *"Dựa trên toàn bộ 6 nguồn tài liệu, hãy vẽ một bảng so sánh chi tiết đường đi của 1 câu gồm 10 từ trong 3 tình huống: (1) Pretrain, (2) LoRA SFT, và (3) Inference (Prefill + Decode). Chỉ rõ ở mỗi tình huống: ma trận nào được đọc, ma trận nào cập nhật, KV Cache và Gradient xuất hiện ở đâu?"*
2. *"Giải thích câu hỏi: Số lượng chuyên gia MoE có phải được ấn định từ Step 0 không? Nếu có, 'chuyên gia' thực chất được hình thành bằng cơ chế toán học nào trong quá trình dữ liệu chảy vào?"*
3. *"So sánh 4 hướng giải quyết bài toán KV Cache trong Inference: MHA, MQA, GQA (`8Q:2KV`) và MLA (DeepSeek-V2/V3). Giải thích vì sao GQA lại quan trọng cả ở pha Rollout của GRPO."*
4. *"Phân tích sự chuyển dịch từ Aux Loss (`seq_aux`) → Router Z-Loss / Capacity Factor → Auxiliary-Loss-Free Dynamic Bias của DeepSeek-V3. Mỗi bước tiến hóa giải quyết vấn đề phụ nào?"*

### B. Prompt Phản biện & Thử thách Ngộ nhận
5. *"Nêu 5 ngộ nhận phổ biến nhất về Dense vs. MoE (ví dụ: về VRAM, về tốc độ LoRA, về việc gán chủ đề cho Expert, về thí nghiệm trên tập dữ liệu nhỏ) và dẫn chứng cụ thể từ tài liệu để phản bác."*
6. *"Nếu một bạn mới hỏi: 'Tại sao bài test 64 câu trong Đấu trường Live cho thấy MoE chưa thắng Dense-18 nhưng lại bảo MoE mạnh hơn Dense?' — hãy giải thích dựa trên cả tỷ lệ tham số Embedding/LM Head (91%) lẫn Chinchilla Scaling Law."*
7. *"Tại sao lượng tử hóa trọng số xuống 4-bit (AWQ/GPTQ) không làm giảm số phép tính FLOPs nhưng vẫn tăng tốc độ sinh token (Decode) lên gấp 2–3 lần?"*

### C. Prompt Tự kiểm tra (Active Recall Quiz)
8. *"Hãy tạo 10 câu hỏi trắc nghiệm và tự luận tình huống (kèm đáp án giải thích) kiểm tra hiểu biết về Residual Stream, Pre-RMSNorm, Prompt Loss Masking (`ignore_index=-100`), AdamW 12B/param, Load Collapse `50/50/0/0` và Temperature Sampling."*

### D. Prompt tạo Audio Overview / Mind Map (Studio của NotebookLM)
9. *Custom Instructions cho Audio Overview:* *"Hãy tổ chức một buổi thảo luận phản biện bằng tiếng Việt (hoặc song ngữ giữ thuật ngữ gốc) giữa 2 kỹ sư AI: một người chuyên về tối ưu phần cứng & Serving cho mô hình Dense trên GPU nhỏ, một người thiết kế cụm train MoE quy mô lớn như DeepSeek-V3. Tập trung vào đánh đổi Compute vs. VRAM, cách chuyên gia MoE tự hình thành từ Step 0, và sự khác biệt giữa Training với Inference."*
