# MiniMind 3D Training & Architecture Visualizer — Implementation Plan

Dự án trực quan hóa 3D theo thời gian thực cho kiến trúc và vòng đời huấn luyện mô hình ngôn ngữ nhỏ (dựa trên kiến trúc **MiniMind**: `Dense Transformer`, `MoE (Mixture of Experts)`, và chuỗi huấn luyện `Pretrain → Full SFT → LoRA → RL (DPO / GRPO)`).

---

## 1. Mục tiêu & Phạm vi
- **Đối tượng người xem:** Cả người **không có kiến thức AI** lẫn người làm kỹ thuật. Mọi lời giải thích có 2 mức, chuyển bằng công tắc `🌱 Dễ hiểu (Đời thường)` ↔ `🔬 Chuyên sâu (Kỹ thuật)`.
- **Mô hình mô phỏng:** Theo thông số **MiniMind2** (bản cũ của repo MiniMind):
  - **Dense — MiniMind2-Small:** `hidden_size = 512`, 8 layers, `vocab_size = 6400` → **25.8M** tham số.
  - **MoE — MiniMind2-MoE:** `hidden_size = 640`, 8 layers, **4 Routed Experts (Top-2) + 1 Shared Expert** → **145M** tổng / **45.2M** kích hoạt mỗi token.
  - Bản hiện tại của repo là **MiniMind-3** (Dense 64M; MoE 198M/64M kích hoạt, 4 experts Top-1, **không** có Shared Expert, GQA 8Q:4KV). Việc chuyển app sang MiniMind-3 chưa được quyết định.
- **Trực quan hóa Kiến trúc 3D (Three.js):**
  - **Dense Transformer:** Token Embedding (`vocab_size = 6400`), **RoPE** (kim đồng hồ chỉ theo vị trí `m` của từ trong câu; góc `m·θ` chỉ phụ thuộc vị trí, không đổi theo step), **GQA 8Q : 2KV** (giảm 75% KV Cache — lợi ích chủ yếu khi suy luận và khi sinh mẫu Rollout của GRPO, không phải khi train), SwiGLU FFN.
  - **MoE:** Router chọn **Top-2 Routed Experts** phát sáng + **1 Shared Expert** luôn hoạt động, kèm cột 3D hiển thị tải từng expert (Expert Load & Aux Loss). Router theo đúng `MoEGate` của MiniMind2: `softmax(gate(x))` → Top-2 → chuẩn hóa lại 2 trọng số (`norm_topk_prob=True`). **Mỗi tầng có Router riêng**, nên cùng một từ có thể đi vào cặp chuyên gia khác nhau ở từng tầng. Mỗi expert có một màu cố định, dùng chung cho 3D, sơ đồ soi tầng và biểu đồ tải.
- **Nhịp 1 step huấn luyện (~4.8 giây ở tốc độ 1x, chỉnh được 0.25x – 3x):**
  1. **Forward — "Đọc cả câu":** Cả 10 từ của câu đi **song song** qua `L0 → L7` (không phải lần lượt từng từ).
  2. **Loss — "Chấm độ bất ngờ":** Tại đầu ra, từ được chấm Loss sáng xanh kèm vòng sóng; từ không chấm Loss (prompt) hiện màu xám.
  3. **Backward — "Vặn núm":** Sóng đạo hàm quét ngược `L7 → L0`; tầng nào có trọng số đang mở khóa thì nháy sáng khi sóng đi qua.
- **Trực quan hóa 4 Giai đoạn Huấn luyện (`Pretrain → SFT → LoRA → RL`):**
  1. **Pretrain:** 100% trọng số mở khóa; Loss được tính trên mọi vị trí token.
  2. **Full SFT — Prompt Loss Masking:** Câu hỏi `User` và câu trả lời `Assistant` đi qua mạng **cùng lúc**. Token `User` có nhãn `ignore_index = -100` nên **không tạo Loss** ở đầu ra, nhưng đạo hàm từ phần `Assistant` **vẫn truyền ngược qua** các vị trí `User` (chúng là ngữ cảnh trong Attention).
  3. **LoRA (Rank-8):** Toàn bộ mạng gốc $W_0$ chuyển sang **Frozen Glass 🔒**. Giống `apply_lora()` của MiniMind, LoRA chỉ gắn vào các lớp `Linear` vuông (`q_proj`, `o_proj`) → **0.13M** tham số train (0.51%) ở Dense, **0.16M** (0.11%) ở MoE. Backward hiện **2 sóng**: sóng mờ đi qua mạng gốc (đạo hàm kích hoạt vẫn lan truyền, nhưng không cập nhật trọng số) và sóng tím đi qua nhánh $A \times B$. Lợi ích chính là **tiết kiệm VRAM** (không cần Gradients và trạng thái AdamW cho $W_0$); tốc độ chỉ tăng nhẹ (~12–15%) vì vẫn phải chạy Forward/Backward qua toàn mạng.
  4. **RL / Alignment (DPO & GRPO)** — cả hai đều bắt đầu từ mô hình sau SFT:
     - **DPO:** **Policy Model** (đang train) cạnh **Reference Model** (đóng băng), cặp đáp án $y_{win}$ (xanh lá) / $y_{lose}$ (đỏ), Margin và dây neo **KL-Divergence**.
     - **GRPO:** 1 Prompt sinh $G = 4$ câu trả lời ($o_1 \dots o_4$), chuẩn hóa điểm thưởng trong nhóm $A_i = (r_i - \mu)/\sigma$, không cần mô hình Critic.
- **Bảng phân rã VRAM (5 phần):** Trọng số bf16 (2 B/tham số) • Gradients bf16 (2 B/tham số train) • AdamW (12 B/tham số train: FP32 master + m + v) • **Reference Model π_ref** (2 B/tham số, chỉ có ở DPO/GRPO) • Kích hoạt.
- **Tính năng cho người mới:**
  - **① Mô hình làm gì?** — Ô đoán chữ tiếp theo với Top-5 xác suất. Loss của từ mẫu = "độ bất ngờ" $-\ln P$, tách riêng với Loss trung bình cả batch. Xác suất tăng liên tục qua các giai đoạn, không tụt khi sang giai đoạn mới.
  - **Cùng 1 câu hỏi — Câu trả lời tiến hóa:** Câu trả lời mẫu cho cùng một câu hỏi qua từng giai đoạn; click vào để chuyển cảnh 3D sang giai đoạn đó.
  - **Story Mode (8 cảnh):** Đi lần lượt từ "Đoán chữ tiếp theo" → Từ điển & RoPE → GQA → MoE → Pretrain → SFT → LoRA → RL. Mỗi cảnh tự chuyển giai đoạn/kiến trúc và đưa camera tới đúng bộ phận đang giải thích.
  - **Thẻ HUD thu gọn được:** Ở chế độ Dễ hiểu, các thẻ chi tiết tự thu gọn để phần 3D rộng rãi.
  - **🧪 Train Thật (Web Worker):** Mạng nơ-ron siêu nhỏ train thật ngay trên trình duyệt (xem Bước 6).
  - **🔍 Soi 1 tầng MoE** và **⚖️ Dense hay MoE — đánh đổi gì?** (xem Bước 8).
- **Nguồn dữ liệu:** Simulator (mặc định), hoặc số liệu train thật từ PyTorch qua hook (`PYTORCH LIVE`). Ở chế độ live, chỉ loss, learning rate, step, grad norm từng tầng, số tham số và tổng VRAM là thật; hoạt ảnh, ô đoán chữ, cách chọn expert và câu trả lời mẫu vẫn là giả lập.

---

## 2. Kiến trúc Hệ thống Kỹ thuật

```text
D:\Projects\ClaudeDemo\AI-Visualize\
├── PLAN.md                        # Bản kế hoạch kiến trúc & triển khai
├── package.json                   # Workspace root (scripts chạy cả server & client)
├── python_hook/
│   ├── minimind_telemetry.py      # Hook gắn vào train_*.py của MiniMind: gửi số liệu train một chiều (PyTorch → app)
│   └── demo_pytorch_stream.py     # Script giả lập 1 lần train chậm để kiểm tra chế độ PYTORCH LIVE
├── packages/
│   ├── shared/src/
│   │   ├── index.ts               # Kiểu dữ liệu dùng chung (TelemetryFrame, VramBreakdown, ControlCommand, ...)
│   │   └── simulator.ts           # MiniMindSimulator — dùng chung cho cả server và client (fallback khi không có server)
│   ├── server/src/
│   │   └── index.ts               # Node.js: WebSocket phát frame mỗi 120ms (~8Hz) + POST /api/telemetry nhận số liệu PyTorch
│   │                              # (giữ chế độ live 30 giây sau gói tin cuối, đổi bằng biến LIVE_TIMEOUT_MS)
│   └── client/                    # Frontend (Vite + TypeScript + Three.js)
│       ├── index.html             # Canvas 3D + HUD 2D (topbar, 2 cột thẻ, thanh Story Mode)
│       ├── vite.config.ts         # Tách `three` thành chunk riêng (three-vendor)
│       └── src/
│           ├── main.ts            # Kết nối WebSocket, đồng bộ nút điều khiển, HUD, Story Mode, nội dung Dễ hiểu / Chuyên sâu
│           ├── expertColors.ts    # Màu cố định của 4 routed expert + shared (dùng chung 3D / SVG / biểu đồ)
│           ├── ui/
│           │   ├── moeXray.ts     # Panel "Soi 1 tầng MoE": sơ đồ SVG rẽ nhánh, công thức, routing 8 tầng, thí nghiệm Aux Loss
│           │   └── archCompare.ts # Thẻ so sánh 3 cột: Dense nhỏ / MoE / Dense to (giả định)
│           ├── scene/
│           │   └── MiniMindScene.ts  # Scene Three.js: các khối 3D, hoạt ảnh 3 pha 60 FPS, camera bay theo giai đoạn, click để soi
│           ├── workers/
│           │   └── microLmWorker.ts  # Micro-LM train thật trong Web Worker (Backprop + Adam)
│           └── styles.css
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
6. **Bước 6 (Nâng cao — Chế độ Thực chiến & Tối ưu):**
   - **End-to-End PyTorch Live Stream Test (`python_hook/demo_pytorch_stream.py`):** Script Python giả lập vòng lặp huấn luyện chậm (gửi `POST /api/telemetry` mỗi 2–3 giây) để kiểm chứng trạng thái `PYTORCH LIVE`, timeout 30 giây, biểu đồ Loss thật và thanh phân rã VRAM.
   - **Live Micro-LM (Train thật trên trình duyệt bằng Web Worker):** Tích hợp mạng nơ-ron ngôn ngữ siêu nhỏ (`Embedding + Hidden Tanh + Linear LM Head`, huấn luyện bằng Backpropagation + Adam thật bằng `Float32Array`) trên tập ~150 câu tiếng Việt chạy trong Web Worker (`packages/client/src/workers/microLmWorker.ts`). Hiển thị Top-5 xác suất Softmax thật, đường Loss thật và ô gõ câu thử nghiệm trực tiếp từ trọng số đang train.
   - **Tối ưu Bundle Size (`vite.config.ts`):** Tách thư viện `three` sang `manualChunks` (`three-vendor`) để giảm kích thước chunk ứng dụng chính và loại bỏ cảnh báo >500kB của Vite.
7. **Bước 7 (Tương tác thật với MiniMind — Inference Server):**
   - **Mục tiêu:** Cho người xem gõ câu hỏi và xem chính mô hình MiniMind trả lời, kèm xác suất từ tiếp theo thật. Ô ① "Đoán chữ tiếp theo" và ô "Cùng 1 Câu hỏi — Câu trả lời tiến hóa" dùng dữ liệu thật thay cho dữ liệu viết sẵn trong `simulator.ts`.
   - **Phân biệt với hook hiện có:** `minimind_telemetry.py` chỉ gửi số liệu train **một chiều** (PyTorch → app). Inference Server là chiều ngược lại: app gửi câu hỏi → MiniMind trả lời.
   - **Server Python nhỏ (`python_hook/minimind_infer_server.py`):** Nạp checkpoint MiniMind (`.pth`) cùng tokenizer 6400 từ của MiniMind, chạy `torch.no_grad()`. Có thể nạp nhiều checkpoint theo giai đoạn (`pretrain`, `full_sft`, `lora`, `dpo`/`grpo`) để so sánh cùng một câu hỏi qua từng giai đoạn.
   - **API dự kiến:**
     - `POST /api/infer/next-token` `{ prompt, stage }` → Top-5 token kế tiếp kèm xác suất Softmax.
     - `POST /api/infer/generate` `{ prompt, stage, max_new_tokens, temperature }` → câu trả lời sinh ra (dùng chat template của MiniMind cho các giai đoạn từ SFT trở đi).
   - **Tích hợp client:** Thêm ô nhập câu hỏi. Nếu Inference Server đang chạy thì gọi API và hiển thị nhãn `MINIMIND THẬT`. Nếu không thì quay về dữ liệu viết sẵn (giống cơ chế fallback của simulator), để app vẫn chạy được khi không có GPU hoặc checkpoint.
   - **Lưu ý:** Phải ghi rõ trên giao diện checkpoint và kiến trúc nào đang được nạp (MiniMind2 hay MiniMind-3, Dense hay MoE), vì câu trả lời phụ thuộc vào checkpoint cụ thể.
8. **Bước 8 (Hiểu "tại sao MoE" — đã làm):**
   - **Thẻ ⚖️ Dense hay MoE — đánh đổi gì? (3 cột):** Dense nhỏ 25.8M • MoE 145M (45.2M kích hoạt) • **Dense to 145M (giả định)**.
     - Cần cột thứ 3 vì Dense nhỏ (`d=512`) và MoE (`d=640`) khác kích thước. So trực tiếp 25.8M với 45.2M sẽ cho thấy MoE tốn sức **hơn** (1.75×), ngược với thông điệp. So với Dense cùng 145M thì MoE chỉ tốn ~1/3.2 công sức mỗi từ.
     - Số liệu lấy từ `MiniMindSimulator.computeFootprint(arch, stage)`, cùng hàm mà `buildFrame()` dùng, nên luôn khớp thẻ VRAM (Dense 734 MB, MoE 2,860 MB ở Pretrain). VRAM của cột giả định chỉ ghi cận dưới `≥`.
     - Khi tải bị sụp đổ, thẻ hiện cảnh báo "chỉ ~2/4 chuyên gia được dùng".
   - **Panel 🔍 Soi 1 tầng MoE** (nút trên topbar, hoặc click vào Router/Expert trong 3D; Story cảnh 4 tự mở ở tầng L2):
     - Tab ① **"1 từ đi qua tầng này"**: chọn từ (rê chuột để xem, click để ghim; không chọn thì tự chuyển mỗi 2.6 giây). Sơ đồ SVG `x → Router → 2 chuyên gia → Σ → y` cộng nhánh Shared đi thẳng không qua Router. Hạt sáng chạy theo 2 nhánh. Bản sao gửi **vào** 2 chuyên gia giống hệt nhau, chỉ **đầu ra** được nhân trọng số, nên đường ra dày theo trọng số và hạt nhỏ lại sau chuyên gia.
     - Công thức sống `y = 0.62×E3(x) + 0.38×E0(x) + Shared(x)`. Dải 8 tầng cho thấy cùng từ đó chọn cặp nào ở mỗi tầng.
     - Nhãn "💤 nghỉ" chỉ áp dụng cho **từng từ**. Bên dưới có bảng phân bổ của cả câu để thấy chuyên gia nào cũng có việc.
     - **Không** gắn nhãn chủ đề kiểu "E0 = địa danh": router thật không chia việc theo chủ đề con người đặt tên được.
     - Khi panel mở, scene 3D cắt bỏ các tầng phía trước, vẽ vòng sáng quanh tầng đang soi và tô routing của từ đang chọn trên cả 8 tầng.
     - Tab ② **"Thí nghiệm Aux Loss"**: công tắc bật/tắt (`SET_AUX_LOSS`), nút ⏩ 3×, biểu đồ tải 4 chuyên gia qua 40 bước, số chuyên gia hiệu dụng `exp(H(load))`.
   - **Mô phỏng sụp đổ tải (trong `simulator.ts`):**
     - Khi tắt Aux Loss, độ lệch tăng dần theo kiểu "giàu càng giàu" (tăng theo hàm logistic, khoảng 15–20 bước). Khi bật lại thì hồi phục dần.
     - Với Top-2, mỗi expert nhận tối đa 50% lượt, nên trạng thái sụp đổ là **50/50/0/0**.
     - Aux loss tính theo `seq_aux` của MiniMind2: `aux = α·Σ ceᵢ·Pᵢ`, `α = 0.1`. Giá trị ≈ 0.10 khi chia đều, ≈ 0.20 khi sụp đổ.
     - Ở **LoRA** Router bị đóng băng nên tải đứng yên.
     - Tải mỗi tầng được ước tính trên một batch ảo 64 token.
   - **Còn mở:** Ở màn hình ~1440×900, panel che phần lớn vùng 3D ở giữa khi đang ở tab ①. Nếu chuyển sang MiniMind-3 (Top-1, không có Shared Expert), cần sửa sơ đồ, công thức và trần tải (Top-1 cho phép 1 expert ôm 100%).
