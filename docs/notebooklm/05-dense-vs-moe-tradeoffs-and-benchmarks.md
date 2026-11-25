# Đánh đổi Thực chiến: Dense vs. MoE (Từ MiniMind2 tới Thực nghiệm Song song)

> So sánh sai cách giữa Dense và MoE rất dễ dẫn đến kết luận ngược. Tài liệu này trình bày **quy tắc so sánh 3 cột công bằng**, ma trận quyết định theo 4 kịch bản thực tế, và **những bài học đo đạc thực nghiệm trung thực** khi chạy song song Dense vs. MoE trên bài toán nhỏ.

---

## 1. Quy tắc 3 Cột: Cách so sánh Dense vs. MoE cho Công bằng

Nếu chỉ đặt hai mô hình thực tế của MiniMind2 cạnh nhau:
- **Dense nhỏ:** `25.8M` tổng / `25.8M` kích hoạt (`hidden = 512`)
- **MoE:** `145.0M` tổng / `45.2M` kích hoạt (`hidden = 640`)

Người xem rất dễ kết luận nhầm: *"MoE tốn công hơn Dense tới `1.75×` (`45.2M / 25.8M`)!"* — lý do chỉ vì hai mô hình có `hidden_size` khác nhau (`512` vs `640`).

Để thấy đúng bản chất của MoE, **luôn phải đặt thêm cột thứ 3: Một mạng Dense to có cùng tổng tham số (`145M`)**:

| Chỉ số | Dense nhỏ (MiniMind2) | MoE (MiniMind2) | Dense to giả định (cùng tổng size) |
| :--- | :---: | :---: | :---: |
| **Tổng "núm vặn" (Sức chứa tri thức)** | `25.8M` | **`145.0M`** | `145.0M` |
| **Núm vặn tính mỗi từ (Active)** | `25.8M` | **`45.2M`** | `145.0M` |
| **FLOPs/từ (so Dense nhỏ)** | `1×` | **`1.75×`** *(nhẹ hơn `3.2×` so Dense to)* | `5.62×` |
| **VRAM Pretrain (`bf16 + AdamW`)** | `734 MB` | **`2,860 MB`** | `≥ 2,860 MB` |

**Kết luận cốt lõi của bảng 3 cột:**
- **Điểm thắng của MoE:** Đạt dung lượng tri thức gần tầm mạng `145M` nhưng chi phí tính toán mỗi token chỉ như mạng `45.2M`.
- **Cái giá phải trả:** GPU vẫn phải chứa toàn bộ `145M` tham số của tất cả các chuyên gia, nên **VRAM gần bằng mạng Dense `145M`**. Nếu Aux Loss bị tắt khiến sụp đổ về ~2/4 expert, ta vẫn trả `2,860 MB` VRAM nhưng chỉ nhận sức chứa của một nửa mô hình.

---

## 2. Bài học Thực nghiệm Trung thực (Kết quả 5 Seed trên Bộ Song song nhỏ)

Trong Đấu trường Live của dự án (`dualArenaCore.ts`), chúng ta cho **3 mô hình thật** cùng chia sẻ khởi tạo ban đầu (`Embedding` và `LM Head`), chạy Backprop + AdamW trên đúng cùng một thứ tự batch đã xáo trộn (`880` mẫu token/epoch từ 64 câu thuộc 4 miền):
- **Dense nhỏ (`18/18` nơ-ron FFN):** `18,840` tổng tham số (`906` FFN)
- **MoE (`4×7 Top-2 + 4 Shared` = `18` active / `32` total FFN):** `19,722` tổng tham số (`1,788` FFN — **chỉ gấp `1.05×` tổng tham số**, riêng FFN gấp `1.97×`)
- **Dense to (`32/32` nơ-ron FFN):** `19,526` tổng tham số (`1,592` FFN — tính gấp `1.78×` nơ-ron FFN mỗi từ)

**Kết quả đo Cross-Entropy Loss ở Epoch 15 (trên 5 seed `1, 7, 42, 123, 999`):**

| Seed | Dense-18 Loss | MoE (Aux BẬT) | `exp(H)` / `ce·P` (BẬT) | MoE (Aux TẮT) | `exp(H)` / `ce·P` (TẮT) | Dense-32 Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `0.1312` | `0.1101` | `3.92` / `1.016` | `0.1433` | `2.58` / `1.399` | `0.0860` |
| `7` | `0.1217` | `0.1225` | `3.99` / `1.006` | `0.1485` | `3.28` / `1.282` | `0.0729` |
| `42` | `0.0835` | `0.1306` | `3.95` / `0.994` | `0.1303` | `2.56` / `1.484` | `0.0741` |
| `123` | `0.1037` | `0.1064` | `3.88` / `0.990` | `0.1196` | `3.32` / `1.188` | `0.0660` |
| `999` | `0.0868` | `0.0993` | `3.97` / `1.000` | `0.1237` | `2.87` / `1.319` | `0.0713` |

### 3 Phát hiện thực nghiệm rất đáng nhớ từ bảng số liệu trên:
1. **Khi kích thước Embedding + LM Head áp đảo FFN (`17,934 / 19,722 ≈ 91%`) trên tập dữ liệu rất nhỏ (`880` token):** FFN không phải nút thắt dung lượng! Lúc này MoE (`18` active, Aux BẬT) đạt CE Loss **gần ngang ngửa** Dense-18 (trung bình ~`0.114` vs `0.105`), còn Dense-32 thắng rõ (`~0.074`) nhờ chạy `32/32` nơ-ron ở mọi bước. **Đừng bao giờ tin các thí nghiệm nhỏ xíu tuyên bố MoE "đập tan" Dense ở iso-FLOPs** — ưu thế MoE chỉ xuất hiện khi FFN chiếm 60–75% mô hình và dữ liệu đủ lớn để làm Dense nhỏ bị quá tải sức chứa.
2. **Tắt Aux Loss làm sụp chuyên gia và tăng Loss thật ngay cả ở mô hình nhỏ:** Khi tắt Aux (`α = 0`), `exp(H)` tụt từ `~3.95/4` xuống còn `2.56–3.32/4`, độ lệch tải `ce·P` tăng vọt lên `1.19–1.48`, và Loss của MoE xấu đi rõ rệt trên hầu hết các seed.
3. **Chuẩn hóa trọng số Top-2 (`norm_topk_prob=True`) và `SHARED_SCALE = 1.0` rất nhạy:** Trước khi sửa, chỉ riêng việc đặt `SHARED_SCALE = 0.2` và không chuẩn hóa gate forward đã làm MoE thua Dense gấp đôi trên mọi seed (`0.24–0.31` vs `0.13–0.17`).

---

## 3. Ma trận Quyết định: Khi nào dùng Dense, khi nào dùng MoE?

| Kịch bản thực tế | Nên chọn | Lý do cốt lõi |
| :--- | :---: | :--- |
| **1. Pretrain trên cụm GPU lớn với dữ liệu hàng nghìn tỷ token** | **MoE** | Tổng chi phí FLOPs để đạt cùng mức Loss thấp hơn nhiều lần so với Dense cùng tổng tham số (tăng tốc train quy mô lớn). |
| **2. Serve API thương mại (nhiều GPU, băng thông lớn, cần tốc độ tok/s cao)** | **MoE** | Mỗi request sinh từ nhanh nhờ kích hoạt tham số nhỏ (`37B/671B` hoặc `45.2M/145M`) trong khi trí thông minh tương đương mô hình lớn hơn nhiều. |
| **3. Chạy Local trên Laptop / Mobile / 1 GPU ít VRAM (`4GB–16GB`)** | **Dense** | VRAM là tài nguyên khan hiếm nhất ở local. Một Dense `7B–8B` lượng tử 4-bit vừa vặn GPU 8GB và tận dụng cache tốt hơn một MoE `8x7B` cần tới `~24GB+` ngay cả ở 4-bit. |
| **4. Fine-tune (SFT/LoRA) trên tập dữ liệu hẹp (< 10K mẫu)** | **Dense** *(hoặc MoE đóng băng Router)* | MoE dễ overfit router hoặc lệch tải trên tập dữ liệu hẹp nếu mở train toàn bộ gate; Dense ổn định và dễ điều chỉnh hơn. |

---

## 4. Ghi chú chuyển tiếp kiến trúc: MiniMind2 vs. MiniMind-3

Trong hệ sinh thái MiniMind có 2 thế hệ đáng lưu ý khi đọc code thật:
- **MiniMind2-MoE (hiện đang mô phỏng):** `hidden = 640`, `4 Routed (Top-2) + 1 Shared Expert`, `norm_topk_prob = True`, `seq_aux α = 0.1`, GQA `8Q:2KV`.
- **MiniMind-3 (bản mới hơn trên repo):** Dense `64M`, MoE `198M` tổng / `64M` active, **`4` experts `Top-1`**, **không có Shared Expert**, GQA `8Q:4KV`. Với `Top-1`, một chuyên gia có thể ôm tới `100%` lượt khi sụp đổ tải (thay vì trần `50%` như `Top-2`).

---

## Câu hỏi gợi ý để hỏi sâu thêm trên NotebookLM

1. *Tại sao quy tắc 3 cột (`Dense nhỏ` vs `MoE` vs `Dense to`) lại cần thiết để tránh ngộ nhận rằng MoE tốn FLOPs hơn Dense?*
2. *Trong thí nghiệm thực tế với 880 token, vì sao MoE chưa thể thắng Dense ở cùng số nơ-ron kích hoạt (18), nhưng Dense-32 lại thấp loss hơn? Điều kiện nào ngoài thực tế đảo ngược cán cân đó?*
3. *So sánh rủi ro sụp đổ tải và mức trần phần trăm mỗi chuyên gia giữa Top-2 (`50%`) ở MiniMind2 và Top-1 (`100%`) ở MiniMind-3.*
4. *Nếu bạn có 1 GPU 12GB VRAM muốn triển khai trợ lý AI cho doanh nghiệp vừa và nhỏ, khi nào bạn chọn Dense 8B Q4 và khi nào bạn cân nhắc MoE?*
