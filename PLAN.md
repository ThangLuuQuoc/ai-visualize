# MiniMind 3D Training & Architecture Visualizer — Implementation Plan

Dự án trực quan hóa 3D theo thời gian thực cho kiến trúc và vòng đời huấn luyện mô hình ngôn ngữ nhỏ (dựa trên kiến trúc **MiniMind**: `Dense Transformer`, `MoE (Mixture of Experts)`, và chuỗi huấn luyện `Pretrain → Full SFT → LoRA → RL (DPO / GRPO)`).

---

## 1. Mục tiêu & Phạm vi
- **Trực quan hóa Kiến trúc 3D (Three.js):**
  - **Dense Transformer:** Token Embedding (`vocab_size = 6400`), RoPE (Rotary Position Embedding xoay góc pha 3D), GQA (Grouped-Query Attention gom nhiều Query heads về ít KV heads), SwiGLU FFN.
  - **MoE (Mixture of Experts):** Cổng Router (Gating Network) phân luồng hạt token sang **Top-K Routed Experts** phát sáng + **1 Shared Expert** luôn hoạt động, kèm biểu đồ cột 3D hiển thị tải phân bổ (Expert Load & Aux Loss).
- **Trực quan hóa 4 Giai đoạn Huấn luyện (`Pretrain → SFT → LoRA → RL`):**
  1. **Pretrain:** Toàn bộ 100% khối trọng số mở khóa (Active); hạt Forward (xanh) và sóng Gradient Backward (đỏ/cam) chảy qua toàn mạng.
  2. **Full SFT:** Hiển thị cơ chế **Prompt Loss Masking** — các token thuộc câu hỏi `User` bị tấm chắn Mask chặn ở đầu ra, chỉ các token `Assistant` mới sinh Gradient ngược.
  3. **LoRA (Low-Rank Adaptation):** Toàn bộ khối trọng số gốc $W_0$ chuyển sang hiệu ứng **Frozen Glass (Đóng băng)**; xuất hiện nhánh phụ hình đồng hồ cát $A \times B$ (thắt cổ chai ở rank $r$) nhận 100% luồng Gradient, minh họa lý do tiết kiệm VRAM và tăng tốc độ train.
  4. **RL / Alignment (DPO & GRPO):**
     - **DPO:** Hiển thị song song **Policy Model** (Active) và **Reference Model** (Frozen), hai luồng đáp án $y_{win}$ (xanh lá) và $y_{lose}$ (đỏ), lực kéo-đẩy Margin và dây neo **KL-Divergence**.
     - **GRPO:** 1 Prompt tách thành chùm $G$ nhánh đầu ra ($o_1 \dots o_G$), đi vào khối **Reward & Group Baseline** mà không cần mô hình Critic riêng biệt.

---

## 2. Kiến trúc Hệ thống Kỹ thuật

```text
D:\Projects\ClaudeDemo\AI-Visualize\
├── PLAN.md                     # Bản kế hoạch kiến trúc & triển khai
├── package.json                # Workspace root (scripts chạy cả server & client)
├── python_hook/
│   └── minimind_telemetry.py   # Hook Python nhẹ tích hợp vào train_*.py của MiniMind
├── packages/
│   ├── shared/                 # TypeScript interfaces dùng chung (TelemetryFrame, StageConfig)
│   ├── server/                 # Node.js + TypeScript WebSocket & REST Server
│   │   ├── src/
│   │   │   ├── simulator.ts    # Bộ phát dữ liệu mô phỏng toán học theo từng Stage & Architecture
│   │   │   └── index.ts        # WebSocket streamer (30Hz) + HTTP endpoint nhận telemetry từ PyTorch
│   └── client/                 # Frontend (Vite + TypeScript + Three.js)
│       ├── index.html          # Giao diện 3D Canvas + 2D HUD Dashboard
│       └── src/
│           ├── main.ts         # Khởi tạo ứng dụng, kết nối WebSocket & Fallback Simulator
│           ├── scene/          # Three.js Scene, Post-processing, InstancedMesh Builders
│           └── ui/             # HUD điều khiển Stage (Pretrain/SFT/LoRA/RL), Arch (Dense/MoE), Charts
```

---

## 3. Các bước triển khai
1. **Bước 1:** Lưu `PLAN.md` và khởi tạo cấu trúc monorepo TypeScript (`shared`, `server`, `client`).
2. **Bước 2:** Xây dựng `packages/shared` định nghĩa schema cho `TrainingStage` (`pretrain` | `sft` | `lora` | `rl_dpo` | `rl_grpo`), `ArchitectureType` (`dense` | `moe`), và `TelemetryFrame`.
3. **Bước 3:** Xây dựng `packages/server` (Node.js + `ws`) vừa chạy **Interactive Simulation Engine** vừa mở cổng HTTP `/api/telemetry` cùng script mẫu `python_hook/minimind_telemetry.py`.
4. **Bước 4:** Xây dựng `packages/client` với **Three.js + TypeScript**:
   - Không gian 3D tương tác (OrbitControls, Raycaster click-to-inspect layer, animations cho Forward/Backward particles, MoE Top-K routing beams, LoRA frozen glass & bypass adapter, DPO/GRPO multi-branch arena).
   - Bảng điều khiển HUD 2D trực quan: Chuyển đổi tức thì giữa `Dense ↔ MoE` và `Pretrain → SFT → LoRA → RL`, biểu đồ Loss / Grad Norm / Expert Load / VRAM Comparison theo thời gian thực.
5. **Bước 5:** Cài đặt dependencies, kiểm tra build TypeScript và khởi chạy hệ thống.
