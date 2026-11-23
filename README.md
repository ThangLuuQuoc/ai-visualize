# MiniMind 3D Training & Architecture Visualizer

Hệ thống trực quan hóa 3D theo thời gian thực (**TypeScript + Three.js + Node.js**) giúp quan sát kiến trúc và cơ chế tối ưu tốc độ huấn luyện của mô hình ngôn ngữ nhỏ (dựa trên kiến trúc **MiniMind**).

## Tính năng chính
- **Kiến trúc 3D (`Dense 64M` ↔ `MoE 198M / 64M Active`):**
  - Bảng từ điển thu gọn (`vocab_size = 6400`) & vòng xoay góc pha **RoPE (Rotary Position Embedding)**.
  - **GQA (Grouped-Query Attention 8Q : 2KV)** tiết kiệm 75% băng thông bộ nhớ KV Cache.
  - **MoE (Mixture of Experts):** Cổng Router kích hoạt **Top-2 Routed Experts** + **1 Shared Expert**, hiển thị cột tải phân bổ 3D và `Aux Loss`.
- **Chuỗi Giai đoạn Huấn luyện (`1. Pretrain` → `2. Full SFT` → `3. LoRA (Rank-8)` → `4a. RL DPO` / `4b. RL GRPO`):**
  - **Pretrain:** 100% trọng số mở khóa, sóng Gradient chạy ngược qua toàn mạng.
  - **Full SFT:** Trực quan hóa tấm chắn **Prompt Loss Masking** (chỉ tính Loss trên câu trả lời của `Assistant`).
  - **LoRA (Rank-8):** Hiệu ứng **Frozen Glass** đóng băng mạng gốc $W_0$ và luồng Gradient tím đi riêng qua nhánh đồng hồ cát $A \times B$ ($r = 8$).
  - **RL (DPO & GRPO):** So sánh cặp $y_{\text{win}}$ / $y_{\text{lose}}$ kèm neo **KL-Divergence** với Frozen Reference Model (DPO), hoặc chùm $G=4$ nhánh tự chuẩn hóa điểm nhóm không cần mô hình Critic (GRPO).
- **Chế độ Kép (Dual-Mode):**
  - Bộ mô phỏng toán học tương tác chạy độc lập hoặc qua Node.js WebSocket (`ws://localhost:4000`).
  - Hỗ trợ gắn hook Python (`python_hook/minimind_telemetry.py`) vào script train PyTorch thực tế của MiniMind.

## Cài đặt & Khởi chạy

```bash
npm install

# Khởi chạy Node.js WebSocket & Telemetry Server (http://localhost:4000)
npm run dev:server

# Khởi chạy giao diện 3D Three.js + Vite (http://localhost:5273)
npm run dev:client
```

Xem chi tiết kế hoạch thiết kế tại [PLAN.md](./PLAN.md).
