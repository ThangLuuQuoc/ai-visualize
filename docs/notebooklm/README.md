# Thư mục Tài liệu Chuyên sâu LLM cho NotebookLM

Thư mục này chứa bộ **6 tài liệu nguồn** được cấu trúc tối ưu để upload trực tiếp vào **Google NotebookLM** (tạo Notebook nghiên cứu, Mind Map, Flashcards, và Audio Overview).

## Danh sách file (theo thứ tự nên đọc / upload)
1. `01-dense-transformer-architecture.md` — Kiến trúc Dense Transformer hiện đại (Residual Stream, Pre-RMSNorm, RoPE, GQA, FlashAttention, SwiGLU).
2. `02-moe-mixture-of-experts-deep-dive.md` — MoE từ toán học đến Mixtral & DeepSeek-V3 (vòng đời Step 0 → Symmetry Breaking → Collapse, Aux Loss vs Dynamic Bias, Fine-grained + Shared Experts, nút thắt phần cứng).
3. `03-training-dynamics-and-post-training.md` — Toàn cảnh Huấn luyện & Tinh chỉnh (Forward/Loss/Backward, phân rã VRAM AdamW `12B/param`, Chinchilla Scaling Laws, Full SFT Prompt Masking, LoRA/QLoRA, PPO → DPO → GRPO).
4. `04-inference-serving-and-sampling.md` — Cách AI Trả về Kết quả (Prefill Compute-bound vs Decode Memory-Bandwidth-bound, KV Cache & MLA, PagedAttention, Temperature/Top-k/Top-p/Min-p, Quantization & Speculative Decoding).
5. `05-dense-vs-moe-tradeoffs-and-benchmarks.md` — Đánh đổi thực chiến Dense vs. MoE (quy tắc 3 cột, kết quả thực nghiệm 5 seed trên Đấu trường song song, ma trận chọn mô hình theo bài toán, MiniMind2 vs MiniMind-3).
6. `06-papers-roadmap-and-study-guide.md` — Bản đồ 14 Paper gốc & 9 bộ prompt mẫu dùng hỏi-đáp / tạo Audio Overview trên NotebookLM.

*(Bạn cũng có thể upload kèm file [../2026-11-25-training-vs-inference-vision.md](../2026-11-25-training-vs-inference-vision.md) nếu muốn NotebookLM nắm luôn định hướng trực quan hóa của app).*

## Cách đưa lên NotebookLM trong 30 giây
1. Mở [https://notebooklm.google.com](https://notebooklm.google.com) → bấm **Create new notebook**.
2. Ở cửa sổ **Add sources**, chọn **Upload sources** (hoặc kéo-thả trực tiếp toàn bộ 6 file `.md` từ `01` đến `06` trong thư mục này).
3. Mở file `06-papers-roadmap-and-study-guide.md` phần cuối để lấy các prompt đào sâu hoặc dùng tính năng **Mind Map / Study Guide / Audio Overview** của NotebookLM.
