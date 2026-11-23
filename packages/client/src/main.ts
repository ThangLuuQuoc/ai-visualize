import {
  ArchitectureType,
  ControlCommand,
  ExplanationLevel,
  MiniMindSimulator,
  TelemetryFrame,
  TrainingStage,
} from '@minimind-viz/shared';
import { InspectableMetadata, MiniMindScene } from './scene/MiniMindScene';
import type { MicroLmUpdateMessage } from './workers/microLmWorker';

interface StageCopy {
  simpleHint: string;
  technicalHint: string;
  simpleHtml: string;
  technicalHtml: string;
}

interface StorySceneStep {
  title: string;
  stage?: TrainingStage;
  arch?: ArchitectureType;
  camPos: [number, number, number];
  camLook: [number, number, number];
  simpleText: string;
  technicalText: string;
}

const STORY_SCENES: StorySceneStep[] = [
  {
    title: '1/8 — Mục tiêu cốt lõi của AI: Đoán chữ tiếp theo',
    stage: 'pretrain',
    arch: 'moe',
    camPos: [18, 12, 36],
    camLook: [0, 0, 26],
    simpleText:
      'Bản chất của các mô hình ngôn ngữ (như ChatGPT hay MiniMind) là trò chơi đoán chữ tiếp theo. Ở thẻ ① bên trái, khi mô hình đoán chữ "Hà_Nội" với xác suất càng cao thì "Độ bất ngờ" (Loss) càng giảm xuống!',
    technicalText:
      'Mô hình tối ưu hàm mục tiêu Autoregressive Cross-Entropy L = -ln P(x_t | x_<t) tại cổng LM Head đầu ra. Đường Loss trung bình giảm từ ~6.85 về ~2.24 khi phân phối xác suất hội tụ.',
  },
  {
    title: '2/8 — Từ điển rút gọn 6.400 từ & Kim vị trí RoPE',
    stage: 'pretrain',
    arch: 'moe',
    camPos: [14, 9, -22],
    camLook: [0, 0, -26],
    simpleText:
      'Nhìn vào vòng tròn có kim đồng hồ vàng ở đầu mạng: Từ điển được rút gọn còn 6.400 từ hay dùng nhất để tra cứu cực nhanh. Kim vàng RoPE chỉ xoay theo vị trí thứ tự của từ trong câu (từ thứ 0, 1, 2...), không đổi theo bước train.',
    technicalText:
      'MiniMind thu nhỏ vocab_size=6400 để giảm >75% tham số ma trận Embedding. RoPE mã hóa vị trí tương đối bằng góc xoay cố định m·θ_j theo chỉ số vị trí m trong chuỗi.',
  },
  {
    title: '3/8 — GQA: 8 nhân viên dùng chung 2 tủ hồ sơ',
    stage: 'pretrain',
    arch: 'dense',
    camPos: [12, 7, -14],
    camLook: [-1, -1.5, -21],
    simpleText:
      'Ở mỗi tầng (khối màu xanh dương bên dưới), thay vì 8 người mỗi người giữ 1 tủ hồ sơ riêng, cả 8 người cùng tra cứu chung 2 tủ hồ sơ (khối nhỏ bên cạnh). Cách này giúp tiết kiệm 75% bộ nhớ lưu trữ khi mô hình sinh câu trả lời!',
    technicalText:
      'Grouped-Query Attention (8 Query Heads chia sẻ 2 Key/Value Heads) giảm 75% dung lượng KV Cache trong giai đoạn suy luận (Inference) và bước sinh mẫu Rollout của GRPO.',
  },
  {
    title: '4/8 — MoE: Lễ tân bệnh viện & 4 Bác sĩ chuyên khoa',
    stage: 'pretrain',
    arch: 'moe',
    camPos: [15, 11, -6],
    camLook: [0, 1.5, -15],
    simpleText:
      'Quả cầu cam là Lễ tân (Router). Với mỗi từ đi vào, Lễ tân chỉ bật đèn gọi đúng 2 trong 4 Bác sĩ chuyên khoa (khối cam sáng) kết hợp với 1 Bác sĩ đa khoa (trụ xanh lá ở giữa). Nhờ vậy mô hình có 145 triệu núm vặn nhưng chỉ tốn sức bằng mạng 45 triệu!',
    technicalText:
      'MiniMind2-MoE gồm 1 Shared Expert + 4 Routed Experts (Top-2 gating). Tổng tham số đạt 145.0M nhưng mỗi token chỉ kích hoạt 45.2M tham số, kèm Aux Loss giữ cân bằng tải.',
  },
  {
    title: '5/8 — Giai đoạn 1 (Pretrain): Đọc hết cả thư viện sách',
    stage: 'pretrain',
    arch: 'moe',
    camPos: [33, 22, 36],
    camLook: [0, 0, 2],
    simpleText:
      'Ở bước Pretrain, mô hình đọc hàng triệu câu văn thô. Nhìn ô "Cùng 1 Câu hỏi" bên phải: sau bước này, nếu bạn đặt câu hỏi, mô hình chỉ biết viết nối tiếp câu hỏi chứ chưa biết trả lời!',
    technicalText:
      'Pretrain cập nhật 100% trọng số trên văn bản tự giám sát. Khi sóng Backward đỏ quét ngược từ L7 về L0, cả khối GQA, Router, Shared Expert và Top-2 Routed Experts đều nháy sáng.',
  },
  {
    title: '6/8 — Giai đoạn 2 (Full SFT): Luyện giải đề Hỏi – Đáp',
    stage: 'sft',
    arch: 'moe',
    camPos: [28, 19, 36],
    camLook: [0, 0, 6],
    simpleText:
      'Cả câu hỏi của bạn (hạt xám) và câu trả lời mẫu (hạt xanh) cùng đi vào mạng song song. Ở cổng đầu ra, mô hình không chấm điểm phần câu hỏi mà chỉ chấm điểm phần trả lời, nhưng sóng kinh nghiệm đỏ vẫn truyền ngược qua cả câu để hiểu ngữ cảnh!',
    technicalText:
      'Full SFT gán nhãn ignore_index=-100 cho các token của User tại LM Head. Tuy nhiên trong pha Backward, đạo hàm từ các token Assistant vẫn lan truyền ngược qua toàn bộ ma trận Causal Attention.',
  },
  {
    title: '7/8 — Giai đoạn 3 (LoRA): Dán giấy ghi chú lên sách bị khóa 🔒',
    stage: 'lora',
    arch: 'moe',
    camPos: [28, 16, 28],
    camLook: [4, 0, 2],
    simpleText:
      'Toàn bộ mạng gốc hóa thành băng trong suốt có gắn ổ khóa 🔒. Một làn sóng mờ vẫn đi xuyên qua mạng gốc để truyền thông tin, nhưng chỉ có các ống đồng hồ cát màu tím (giấy ghi chú LoRA) ở lề phải được vặn núm — giảm hơn một nửa bộ nhớ VRAM!',
    technicalText:
      'LoRA (Rank-8) đóng băng W_0, gắn vào q_proj và o_proj (các lớp Linear vuông d×d), chỉ cập nhật 0.16M tham số (0.11% của 145M MoE, hoặc 0.13M = 0.51% của 25.8M Dense). Tiết kiệm gần như toàn bộ 12 bytes/param của bộ nhớ AdamW Optimizer.',
  },
  {
    title: '8/8 — Giai đoạn 4 (RL DPO & GRPO): So sánh bài văn & Tự chấm nhóm',
    stage: 'rl_grpo',
    arch: 'moe',
    camPos: [28, 20, 46],
    camLook: [0, 1, 12],
    simpleText:
      'Cuối cùng, mô hình tự viết nháp 4 bài trả lời (4 khối vuông ở cuối mạng) rồi lấy điểm trung bình của cả 4 bài làm mốc tự thưởng/phạt (GRPO). Không cần thuê thêm mô hình Giám khảo riêng, câu trả lời trở nên lịch sự và chặt chẽ!',
    technicalText:
      'GRPO lấy mẫu G=4 hoàn thành (o_1..o_4) và chuẩn hóa điểm thưởng tương đối A_i = (r_i - μ)/σ. Trên thanh VRAM, bản sao Reference Model (π_ref) được tách riêng thành đoạn màu xanh băng.',
  },
];

const STAGE_COPY: Record<TrainingStage, StageCopy> = {
  pretrain: {
    simpleHint:
      'Cả câu 10 từ cùng chạy song song vào mạng. Ở Pretrain, mô hình tập đoán chữ tiếp theo ở mọi vị trí trong câu.',
    technicalHint:
      'Pretrain (Causal LM): Đưa song song cả chuỗi [B, L] qua mạng; tính Cross-Entropy Loss trên tất cả các vị trí token.',
    simpleHtml: `
      <b>📚 Ẩn dụ: Đọc hết cả thư viện sách</b><br/>
      • Mô hình chưa biết ai đang hỏi hay trả lời, chỉ đọc hàng triệu câu văn để học quy luật tiếng Việt và kiến thức nền.<br/>
      • Khi đoán sai chữ tiếp theo, sóng đỏ chạy ngược về để chỉnh lại <b>100% các núm vặn</b> trong mạng.
    `,
    technicalHtml: `
      <b>🔬 Cơ chế Pretrain & Cấu hình MiniMind2 gốc</b><br/>
      • <b>MiniMind2-Small (Dense):</b> <code>hidden=512, 8 layers, vocab=6400</code> → <b>25.8M</b> tham số.<br/>
      • <b>MiniMind2-MoE:</b> <code>hidden=640, 8 layers, 4 Routed + 1 Shared</code> → <b>145.0M</b> tổng tham số, chỉ kích hoạt <b>45.2M</b> mỗi token.
    `,
  },
  sft: {
    simpleHint:
      'Cả câu hỏi và câu trả lời vẫn đi vào mạng cùng lúc! Phần câu hỏi dùng làm ngữ cảnh (không chấm điểm), chỉ chấm điểm phần trả lời.',
    technicalHint:
      'Full SFT: Xử lý song song cả chuỗi. Nhãn phần User đặt ignore_index=-100 (không phát sinh Loss tại đầu ra đó), nhưng đạo hàm từ Assistant vẫn lan truyền ngược qua ngữ cảnh User.',
    simpleHtml: `
      <b>📝 Ẩn dụ: Luyện giải đề Hỏi – Đáp theo mẫu</b><br/>
      • Đề bài (User) và bài giải (Assistant) được đọc cùng lúc.<br/>
      • Thầy giáo <b>không chấm điểm phần chép lại đề bài</b> (hạt xám), mà chỉ chấm điểm phần trả lời rồi truyền kinh nghiệm ngược về toàn mạng.
    `,
    technicalHtml: `
      <b>🔬 Cơ chế Prompt Loss Masking chuẩn xác</b><br/>
      • Toàn bộ chuỗi <code>[User, Assistant]</code> đi qua Forward trong 1 lần chạy song song.<br/>
      • Tại đầu ra LM Head, các token <code>User</code> bị mask (<code>loss = 0</code>). Tuy nhiên trong pha Backward, Gradient từ các token <code>Assistant</code> vẫn truyền ngược xuyên qua các lớp Attention của cả chuỗi.
    `,
  },
  lora: {
    simpleHint:
      'Mạng gốc bị khóa băng 🔒 (vẫn có sóng đạo hàm mờ đi qua để truyền thông tin, nhưng chỉ vặn núm tờ giấy ghi chú màu tím A × B ở lề phải!).',
    technicalHint:
      'LoRA (Rank-8): Đóng băng W_0 (không lưu Gradients 2B và AdamW 12B cho W_0). Đạo hàm kích hoạt dL/dx vẫn truyền qua mạng gốc nhưng chỉ cập nhật nhánh A × B.',
    simpleHtml: `
      <b>📌 Ẩn dụ: Dán giấy ghi chú lên sách thay vì viết lại cả cuốn</b><br/>
      • Cuốn sách gốc bị khóa ổ khóa <b>🔒 Đóng băng</b> (sóng mờ đi qua không làm thay đổi sách gốc).<br/>
      • Ta chỉ chỉnh tờ ghi chú mỏng ở lề phải (<b>0.51%</b> số núm vặn ở bản Dense 25.8M, hoặc <b>0.11%</b> ở bản MoE 145M), cắt giảm hoàn toàn bộ nhớ nháp AdamW khổng lồ!
    `,
    technicalHtml: `
      <b>🔬 Vì sao LoRA tiết kiệm VRAM hơn là tăng tốc độ tok/s?</b><br/>
      • <b>Hai sóng Backward trên hình 3D:</b> Sóng mờ ở trục giữa là đạo hàm kích hoạt (<code>dL/dh</code>) vẫn phải truyền qua mạng gốc; sóng tím sáng ở lề phải là cập nhật trọng số <code>dL/dA, dL/dB</code>.<br/>
      • <b>Tiết kiệm VRAM:</b> Giảm vùng màu cam (AdamW 12 bytes/param: FP32 master + m + v) từ <code>1,740 MB</code> xuống chỉ còn <code>~8 MB</code>!
    `,
  },
  rl_dpo: {
    simpleHint:
      'Thầy giáo đưa 2 bài văn: 1 bài hay (cầu xanh lá) và 1 bài dở (cầu đỏ), dạy mô hình ưu tiên bài hay và giữ dây neo với bản gốc.',
    technicalHint:
      'DPO: Tối ưu trực tiếp hiệu số log-ratio giữa đáp án thắng (y_win) và thua (y_lose) so với mô hình tham chiếu đóng băng π_ref (đoạn màu xanh băng trên thanh VRAM).',
    simpleHtml: `
      <b>⚖️ Ẩn dụ: Thầy giáo so sánh 2 bài văn mẫu</b><br/>
      • Thay vì chấm điểm từng chữ, thầy đưa cặp bài <b>Hay (y_win)</b> và <b>Dở (y_lose)</b>.<br/>
      • Mô hình tự chỉnh núm vặn để tăng xác suất viết giống bài Hay, đồng thời có sợi dây neo với <b>Bản gốc đóng băng (π_ref)</b> bên trái để không bị nói năng kỳ quặc.
    `,
    technicalHtml: `
      <b>🔬 Direct Preference Optimization (DPO)</b><br/>
      • Biến đổi nghiệm giải tích của bài toán RL có ràng buộc KL thành hàm mất mát trực tiếp trên cặp <code>(y_win, y_lose)</code>.<br/>
      • Trên thanh phân rã VRAM, <b>Bản gốc π_ref (2 bytes/param)</b> được tách riêng thành đoạn màu xanh băng (<code>290 MB</code> với MoE / <code>52 MB</code> với Dense).
    `,
  },
  rl_grpo: {
    simpleHint:
      'Với 1 câu hỏi, mô hình tự viết 4 bài trả lời nháp rồi lấy điểm trung bình của 4 bài làm mốc tự thưởng/phạt — không tốn bộ nhớ thuê Giám khảo riêng!',
    technicalHint:
      'GRPO: Lấy mẫu nhóm G=4 câu trả lời (o_1..o_4), chuẩn hóa điểm thưởng trong nhóm A_i = (r_i - mean)/std làm baseline, loại bỏ hoàn toàn mô hình Critic của PPO.',
    simpleHtml: `
      <b>🏆 Ẩn dụ: Tự làm 4 bài nháp rồi so điểm trong nhóm</b><br/>
      • Ở cách cũ (PPO), phải nuôi thêm một ông Giám khảo khổng lồ (Critic) rất tốn bộ nhớ.<br/>
      • Với <b>GRPO</b>, mô hình tự viết 4 câu trả lời khác nhau cho cùng 1 đề, tính điểm trung bình của cả 4. Bài nào trên trung bình (xanh) được khuyến khích, dưới trung bình (đỏ) bị giảm đi!
    `,
    technicalHtml: `
      <b>🔬 Group Relative Policy Optimization (GRPO)</b><br/>
      • Thay thế Value/Critic Network của PPO bằng thống kê tương đối của nhóm mẫu <code>G = 4</code> hoàn thành.<br/>
      • Ở bước sinh <code>G</code> câu trả lời (Rollout), cơ chế <b>GQA (8Q : 2KV)</b> giảm 75% bộ nhớ KV Cache của cả 4 nhánh.
    `,
  },
};

const ANALOGY_CHEATSHEET_HTML = `
  <div class="analogy-item"><b>🏥 MoE:</b> Lễ tân chuyển mỗi từ cho 2 bác sĩ chuyên khoa + 1 bác sĩ đa khoa.</div>
  <div class="analogy-item"><b>🗄️ GQA:</b> 8 nhân viên dùng chung 2 tủ hồ sơ (nhẹ bộ nhớ khi sinh chữ).</div>
  <div class="analogy-item"><b>🧭 RoPE:</b> Kim đồng hồ chỉ phụ thuộc vị trí từ m trong câu (m × θ).</div>
  <div class="analogy-item"><b>📉 Loss:</b> Độ "bất ngờ" khi đoán chữ (từ mẫu) vs Trung bình cả cuốn sách.</div>
`;

const TECHNICAL_CHEATSHEET_HTML = `
  <div class="analogy-item"><b>⚡ MoE Top-2:</b> Kích hoạt 45.2M / 145M tham số mỗi token + Aux Loss.</div>
  <div class="analogy-item"><b>🗄️ GQA 8Q:2KV:</b> Giảm 75% KV Cache khi Inference & GRPO Rollout.</div>
  <div class="analogy-item"><b>🧭 RoPE:</b> Góc xoay cố định m·θ_j theo vị trí từ m (bất biến theo step).</div>
  <div class="analogy-item"><b>💾 AdamW 12B:</b> FP32 master (4B) + FP32 m (4B) + FP32 v (4B).</div>
`;

class MiniMindVisualizerApp {
  private scene: MiniMindScene;
  private localSim = new MiniMindSimulator();
  private ws: WebSocket | null = null;
  private wsConnected = false;
  private explanationLevel: ExplanationLevel = 'simple';
  private lossHistory: number[] = [];
  private selectedMeta: InspectableMetadata | null = null;
  private latestFrame: TelemetryFrame | null = null;

  // Cache keys to avoid unnecessary innerHTML replacement (prevents lost clicks & hover flicker!)
  private lastInsightRenderKey = '';
  private storyActive = true;
  private storyIdx = 0;

  // Real In-Browser Neural Micro-LM Web Worker state
  private microLmWorker: Worker | null = null;
  private microLmActive = false;
  private microLmPaused = false;
  private latestMicroLm: MicroLmUpdateMessage | null = null;

  constructor() {
    const canvasContainer = document.getElementById('canvas-container')!;
    const labelsContainer = document.getElementById('labels-container')!;
    this.scene = new MiniMindScene(canvasContainer, labelsContainer);

    this.scene.onNodeSelected((meta, frame) => {
      this.selectedMeta = meta;
      // Auto-expand the 3D Inspector card when the user clicks a 3D object
      this.setCardCollapsed('body-inspect', false);
      this.renderInspector(meta, frame);
    });

    this.bindUIEvents();
    this.renderStoryScene();
    this.connectWebSocket();

    this.onTelemetryFrame(this.localSim.buildFrame(), false);

    setInterval(() => {
      if (!this.wsConnected) {
        const frame = this.localSim.tick(0.12);
        this.onTelemetryFrame(frame, false);
      }
    }, 120);
  }

  private connectWebSocket(): void {
    try {
      const ws = new WebSocket('ws://localhost:4000');
      this.ws = ws;

      ws.onopen = () => {
        this.wsConnected = true;
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'TELEMETRY_FRAME' && msg.frame) {
            this.onTelemetryFrame(msg.frame as TelemetryFrame, true);
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        this.wsConnected = false;
        setTimeout(() => this.connectWebSocket(), 3500);
      };

      ws.onerror = () => {
        this.wsConnected = false;
      };
    } catch {
      this.wsConnected = false;
    }
  }

  private sendCommand(cmd: ControlCommand): void {
    this.localSim.applyCommand(cmd);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    } else {
      this.onTelemetryFrame(this.localSim.buildFrame(), false);
    }
  }

  private setCardCollapsed(targetId: string, collapsed: boolean): void {
    const body = document.getElementById(targetId);
    const trigger = document.querySelector<HTMLElement>(`[data-target="${targetId}"]`);
    if (!body || !trigger) return;
    body.classList.toggle('collapsed', collapsed);
    const caret = trigger.querySelector('.caret');
    if (caret) {
      caret.textContent = collapsed ? '▸' : '▾';
    }
  }

  private bindUIEvents(): void {
    // Collapsible HUD card headers
    document.querySelectorAll<HTMLElement>('.collapsible-trigger').forEach((header) => {
      header.addEventListener('click', () => {
        const targetId = header.dataset.target;
        if (!targetId) return;
        const body = document.getElementById(targetId);
        if (!body) return;
        const willCollapse = !body.classList.contains('collapsed');
        this.setCardCollapsed(targetId, willCollapse);
      });
    });

    // Explanation level toggle (Simple vs Technical)
    const levelBtns = document.querySelectorAll<HTMLButtonElement>('#level-switcher button');
    levelBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        levelBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.explanationLevel = (btn.dataset.level as ExplanationLevel) || 'simple';
        // In Simple mode, collapse secondary detail cards to keep the screen spacious; in Technical mode, expand all
        const isSimple = this.explanationLevel === 'simple';
        this.setCardCollapsed('body-tokens', isSimple);
        this.setCardCollapsed('body-inspect', isSimple);
        this.renderStoryScene();
        if (this.latestFrame) {
          this.onTelemetryFrame(this.latestFrame, this.wsConnected);
        }
      });
    });

    // Architecture buttons
    const archBtns = document.querySelectorAll<HTMLButtonElement>('#arch-switcher button');
    archBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const arch = btn.dataset.arch as ArchitectureType;
        this.sendCommand({ type: 'SET_ARCH', architecture: arch });
      });
    });

    // Stage buttons
    const stageBtns = document.querySelectorAll<HTMLButtonElement>('#stage-switcher button');
    stageBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const stage = btn.dataset.stage as TrainingStage;
        this.lossHistory = [];
        this.sendCommand({ type: 'SET_STAGE', stage });
      });
    });

    // Story Mode controls
    const storyBar = document.getElementById('story-bar')!;
    const btnStoryToggle = document.getElementById('btn-story-toggle')!;
    btnStoryToggle.addEventListener('click', () => {
      this.storyActive = !this.storyActive;
      storyBar.classList.toggle('hidden', !this.storyActive);
      btnStoryToggle.classList.toggle('active', this.storyActive);
      if (this.storyActive) {
        this.applyStorySceneAction();
      }
    });

    document.getElementById('btn-story-prev')!.addEventListener('click', () => {
      this.storyIdx = (this.storyIdx - 1 + STORY_SCENES.length) % STORY_SCENES.length;
      this.applyStorySceneAction();
    });

    document.getElementById('btn-story-next')!.addEventListener('click', () => {
      this.storyIdx = (this.storyIdx + 1) % STORY_SCENES.length;
      this.applyStorySceneAction();
    });

    document.getElementById('btn-story-close')!.addEventListener('click', () => {
      this.storyActive = false;
      storyBar.classList.add('hidden');
      btnStoryToggle.classList.remove('active');
    });

    // Real In-Browser Micro-LM Web Worker Controls
    const btnMicroLm = document.getElementById('btn-microlm-toggle')!;
    const microLmPanel = document.getElementById('microlm-controls')!;
    const promptInput = document.getElementById('microlm-prompt-input') as HTMLInputElement;

    btnMicroLm.addEventListener('click', () => {
      this.microLmActive = !this.microLmActive;
      btnMicroLm.classList.toggle('active', this.microLmActive);
      microLmPanel.classList.toggle('hidden', !this.microLmActive);
      this.lossHistory = [];

      if (this.microLmActive) {
        this.setCardCollapsed('body-pred', false);
        this.ensureMicroLmWorker();
        this.microLmPaused = false;
        this.microLmWorker?.postMessage({ type: 'START' });
      } else {
        this.microLmWorker?.postMessage({ type: 'PAUSE' });
      }
    });

    const applyPrompt = (text: string) => {
      promptInput.value = text;
      this.microLmWorker?.postMessage({ type: 'SET_PROMPT', prompt: text });
    };

    document.getElementById('btn-microlm-apply')!.addEventListener('click', () => {
      applyPrompt(promptInput.value);
    });

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') applyPrompt(promptInput.value);
    });

    document.querySelectorAll<HTMLButtonElement>('.microlm-pills .pill-btn').forEach((pill) => {
      pill.addEventListener('click', () => {
        if (pill.dataset.prompt) applyPrompt(pill.dataset.prompt);
      });
    });

    // Speed slider
    const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
    const speedLabel = document.getElementById('speed-label')!;
    speedSlider.addEventListener('input', () => {
      const speed = Number(speedSlider.value);
      speedLabel.textContent = `${speed.toFixed(2).replace(/\.00$/, '.0')}x`;
      this.sendCommand({ type: 'SET_SPEED', speed });
      this.microLmWorker?.postMessage({ type: 'SET_SPEED', speed });
    });

    document.getElementById('btn-pause')!.addEventListener('click', () => {
      if (this.microLmActive) {
        this.microLmPaused = !this.microLmPaused;
        this.microLmWorker?.postMessage({ type: this.microLmPaused ? 'PAUSE' : 'RESUME' });
      }
      this.sendCommand({ type: 'TOGGLE_PAUSE' });
    });

    document.getElementById('btn-step')!.addEventListener('click', () => {
      if (this.microLmActive) {
        this.microLmPaused = true;
        this.microLmWorker?.postMessage({ type: 'STEP_ONCE' });
      }
      this.sendCommand({ type: 'STEP_ONCE' });
    });

    document.getElementById('btn-reset')!.addEventListener('click', () => {
      this.lossHistory = [];
      if (this.microLmActive) {
        this.microLmWorker?.postMessage({ type: 'RESET' });
      }
      this.sendCommand({ type: 'RESET' });
    });

    document.getElementById('btn-cam-reset')!.addEventListener('click', () => {
      this.scene.resetCamera();
    });
  }

  private renderStoryScene(): void {
    const s = STORY_SCENES[this.storyIdx];
    document.getElementById('story-step-badge')!.textContent =
      `CẢNH ${this.storyIdx + 1} / ${STORY_SCENES.length}`;
    document.getElementById('story-title')!.textContent = s.title;
    document.getElementById('story-desc')!.textContent =
      this.explanationLevel === 'simple' ? s.simpleText : s.technicalText;
  }

  private applyStorySceneAction(): void {
    this.renderStoryScene();
    const s = STORY_SCENES[this.storyIdx];
    // Fly first (and tell the scene to skip its automatic stage camera), then switch stage/arch.
    this.scene.flyCameraForStory(s.camPos, s.camLook, s.stage);
    if (s.arch) {
      this.sendCommand({ type: 'SET_ARCH', architecture: s.arch });
    }
    if (s.stage) {
      this.sendCommand({ type: 'SET_STAGE', stage: s.stage });
    }
  }

  private syncControlsWithFrame(frame: TelemetryFrame): void {
    const stageBtns = document.querySelectorAll<HTMLButtonElement>('#stage-switcher button');
    stageBtns.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.stage === frame.stage);
    });

    const archBtns = document.querySelectorAll<HTMLButtonElement>('#arch-switcher button');
    archBtns.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.arch === frame.architecture);
    });

    const btnPause = document.getElementById('btn-pause')!;
    btnPause.textContent = frame.paused ? '▶ Tiếp tục' : '⏸ Tạm dừng';

    const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
    const speedLabel = document.getElementById('speed-label')!;
    if (document.activeElement !== speedSlider && typeof frame.speed === 'number') {
      speedSlider.value = String(frame.speed);
      speedLabel.textContent = `${frame.speed.toFixed(2).replace(/\.00$/, '.0')}x`;
    }
  }

  private ensureMicroLmWorker(): void {
    if (this.microLmWorker) return;
    const worker = new Worker(new URL('./workers/microLmWorker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (ev: MessageEvent<MicroLmUpdateMessage>) => {
      if (ev.data?.type === 'MICRO_LM_UPDATE') {
        this.latestMicroLm = ev.data;
        if (this.microLmActive && this.latestFrame) {
          this.onTelemetryFrame(this.latestFrame, this.wsConnected);
        }
      }
    };
    this.microLmWorker = worker;
  }

  private mergeMicroLmOverlay(baseFrame: TelemetryFrame): TelemetryFrame {
    const mlm = this.latestMicroLm;
    if (!this.microLmActive || !mlm) return baseFrame;

    const [gE, gW1, gW2] = mlm.gradNorms;
    const layers = baseFrame.layers.map((l, idx) => {
      const realGrad = idx === 0 ? gE : idx < 7 ? gW1 : gW2;
      return {
        ...l,
        baseWeightGradNorm: baseFrame.stage === 'lora' ? 0 : realGrad,
        loraAdapterGradNorm: baseFrame.stage === 'lora' ? realGrad : 0,
      };
    });

    return {
      ...baseFrame,
      step: mlm.step,
      epoch: mlm.epoch,
      loss: mlm.batchLoss,
      layers,
      predictionDemo: {
        promptContext: `Train thật (Web Worker • ${mlm.totalParams.toLocaleString()} params, Vocab ${mlm.vocabSize}): "${mlm.promptText} [ ___ ]"`,
        targetToken: mlm.targetToken,
        targetProb: mlm.targetProb,
        tokenSurpriseLoss: mlm.tokenSurpriseLoss,
        batchAvgLoss: mlm.batchLoss,
        candidates: mlm.candidates,
      },
    };
  }

  private onTelemetryFrame(rawFrame: TelemetryFrame, viaServer: boolean): void {
    this.latestFrame = rawFrame;
    const frame = this.mergeMicroLmOverlay(rawFrame);
    this.syncControlsWithFrame(frame);
    this.scene.updateFromTelemetry(frame);

    const isSimple = this.explanationLevel === 'simple';

    // Update Live Micro-LM generated output box if active
    if (this.microLmActive && this.latestMicroLm) {
      document.getElementById('microlm-gen-output')!.textContent =
        `«${this.latestMicroLm.generatedText}»`;
    }

    // Connection badge
    const badge = document.getElementById('conn-badge')!;
    if (this.microLmActive) {
      badge.textContent = 'WEB WORKER (TRAIN THẬT)';
      badge.className = 'conn-badge worker';
    } else if (frame.source === 'pytorch_live') {
      badge.textContent = 'PYTORCH LIVE';
      badge.className = 'conn-badge live';
    } else if (viaServer) {
      badge.textContent = 'NODE.JS STREAM';
      badge.className = 'conn-badge sim';
    } else {
      badge.textContent = 'LOCAL SIM';
      badge.className = 'conn-badge sim';
    }

    // 1. In-place update for Next-Token Prediction & Surprise Loss Card
    this.renderNextTokenPredictionInPlace(frame, isSimple);

    // 2. In-place update for 3-Phase Stepper & Parallel Sentence Chips
    const phasePill = document.getElementById('phase-pill')!;
    const pFwd = document.getElementById('pstep-forward')!;
    const pLoss = document.getElementById('pstep-loss')!;
    const pBwd = document.getElementById('pstep-backward')!;
    pFwd.className = `pstep ${frame.phase === 'forward' ? 'active forward' : ''}`;
    pLoss.className = `pstep ${frame.phase === 'loss_compute' ? 'active loss' : ''}`;
    pBwd.className = `pstep ${frame.phase === 'backward' ? 'active backward' : ''}`;

    if (frame.phase === 'forward') {
      phasePill.textContent = '1. ĐỌC CẢ CÂU';
    } else if (frame.phase === 'loss_compute') {
      phasePill.textContent = '2. CHẤM ĐỘ BẤT NGỜ';
    } else {
      phasePill.textContent = '3. VẶN NÚM (BACKWARD)';
    }
    phasePill.className = `phase-pill ${frame.phase}`;

    // Only update static stage copy when (stage, level) actually changes
    const insightKey = `${frame.stage}:${this.explanationLevel}`;
    if (this.lastInsightRenderKey !== insightKey) {
      this.lastInsightRenderKey = insightKey;
      const copy = STAGE_COPY[frame.stage];
      document.getElementById('token-stream-hint')!.textContent = isSimple
        ? copy.simpleHint
        : copy.technicalHint;
      document.getElementById('insight-title')!.innerHTML = `<span class="caret">${
        document.getElementById('body-insight')?.classList.contains('collapsed') ? '▸' : '▾'
      }</span> ${isSimple ? '② Giải thích theo Ví dụ Đời thường' : '② Giải thích Cơ chế Kỹ thuật'}`;
      document.getElementById('stage-tag')!.textContent = frame.stage.toUpperCase();
      document.getElementById('optimization-insight')!.innerHTML = isSimple
        ? copy.simpleHtml
        : copy.technicalHtml;
      document.getElementById('analogy-cheatsheet')!.innerHTML = isSimple
        ? ANALOGY_CHEATSHEET_HTML
        : TECHNICAL_CHEATSHEET_HTML;
    }

    // In-place update for Token chips (preserves DOM nodes across ticks!)
    this.renderTokenChipsInPlace(frame, isSimple);

    // 3. In-place update for Stage Output Evolution Comparison Card (never drops clicks or hover!)
    this.renderStageOutputComparisonInPlace(frame);

    // 4. Right panel metrics & 5-Segment VRAM Breakdown
    document.getElementById('step-counter')!.textContent =
      `Step ${frame.step} • Epoch ${frame.epoch}`;
    document.getElementById('m-loss')!.textContent = frame.loss.toFixed(4);
    document.getElementById('m-lr')!.textContent = frame.learningRate.toExponential(2);
    document.getElementById('m-tps')!.textContent =
      `${frame.tokensPerSec.toLocaleString()} tok/s`;
    const laptopGpuPct = Math.round((frame.vramUsedMB / 4096) * 100);
    document.getElementById('m-vram')!.textContent =
      `${frame.vramUsedMB.toLocaleString()} MB (~${laptopGpuPct}% GPU 4GB)`;
    document.getElementById('m-params')!.textContent =
      `${frame.trainableParamsM}M (${frame.trainablePct}%) / ${frame.activeParamsM}M / ${frame.totalParamsM}M`;

    this.renderVramBreakdownInPlace(frame);

    const lastLoss = this.lossHistory[this.lossHistory.length - 1];
    if (lastLoss === undefined || Math.abs(lastLoss - frame.loss) > 0.0005) {
      this.lossHistory.push(frame.loss);
      if (this.lossHistory.length > 50) this.lossHistory.shift();
      this.drawLossChart();
    }

    this.renderDynamicPanelInPlace(frame, isSimple);

    if (this.selectedMeta) {
      this.renderInspector(this.selectedMeta, frame);
    }
  }

  private renderNextTokenPredictionInPlace(frame: TelemetryFrame, isSimple: boolean): void {
    const demo = frame.predictionDemo;
    if (!demo) return;

    const targetPct = Math.round(demo.targetProb * 100);
    document.getElementById('surprise-pill')!.textContent =
      `Độ bất ngờ từ này: ${demo.tokenSurpriseLoss.toFixed(2)}`;
    document.getElementById('pred-context')!.textContent = demo.promptContext;

    const container = document.getElementById('pred-candidates')!;
    // Build candidate rows once, then update in place
    if (container.children.length !== demo.candidates.length) {
      container.innerHTML = demo.candidates
        .map(
          () => `
          <div class="cand-row">
            <span class="cand-word"></span>
            <div class="bar-track"><div class="bar-fill"></div></div>
            <span class="mono-text cand-pct" style="width:36px;text-align:right;"></span>
          </div>
        `
        )
        .join('');
    }

    demo.candidates.forEach((c, idx) => {
      const row = container.children[idx] as HTMLElement;
      const pct = Math.round(c.prob * 100);
      row.className = `cand-row ${c.isTarget ? 'target' : ''}`;
      (row.querySelector('.cand-word') as HTMLElement).textContent =
        `${c.token}${c.isTarget ? ' ✓' : ''}`;
      const fill = row.querySelector('.bar-fill') as HTMLElement;
      fill.style.width = `${Math.max(4, pct)}%`;
      fill.style.background = c.isTarget ? '#34d399' : '#64748b';
      (row.querySelector('.cand-pct') as HTMLElement).textContent = `${pct}%`;
    });

    document.getElementById('pred-explanation')!.textContent = isSimple
      ? `Step ${frame.step}: Tự tin vào "${demo.targetToken}" đạt ${targetPct}% → Độ bất ngờ riêng từ này là ${demo.tokenSurpriseLoss.toFixed(2)} (còn Loss trung bình cả bộ sách ở bên phải là ${frame.loss.toFixed(2)}).`
      : `Loss từ mẫu: -ln(${demo.targetProb}) = ${demo.tokenSurpriseLoss.toFixed(3)} | Loss trung bình cả batch đa dạng: ${frame.loss.toFixed(4)} (kế thừa xác suất liên tục từ Pretrain → SFT → LoRA → RL).`;
  }

  private renderTokenChipsInPlace(frame: TelemetryFrame, isSimple: boolean): void {
    const container = document.getElementById('token-list')!;
    if (container.children.length !== frame.currentTokens.length) {
      container.innerHTML = frame.currentTokens
        .map(
          () => `
          <div class="token-chip">
            <span class="tok-text"></span>
            <span class="token-meta"></span>
          </div>
        `
        )
        .join('');
    }

    frame.currentTokens.forEach((t, idx) => {
      const chip = container.children[idx] as HTMLElement;
      chip.className = `token-chip ${t.lossMasked ? 'masked' : ''}`;
      (chip.querySelector('.tok-text') as HTMLElement).textContent = t.text;
      const routingInfo =
        frame.architecture === 'moe'
          ? `E${t.activeExperts.join(',E')}`
          : `m=${t.positionIdx} (${t.ropeAngleRad}r)`;
      (chip.querySelector('.token-meta') as HTMLElement).textContent = t.lossMasked
        ? isSimple
          ? 'Ngữ cảnh (0 Loss)'
          : `ignore=-100 • m=${t.positionIdx}`
        : `Chấm Loss • ${routingInfo}`;
    });
  }

  private renderStageOutputComparisonInPlace(frame: TelemetryFrame): void {
    document.getElementById('comp-question')!.textContent = frame.comparisonQuestion;
    const container = document.getElementById('stage-answers-list')!;

    // Build DOM elements and attach click listeners ONCE so clicks never slip!
    if (container.children.length !== frame.stageAnswers.length) {
      container.innerHTML = '';
      frame.stageAnswers.forEach((item) => {
        const el = document.createElement('div');
        el.className = 'stage-ans-item';
        el.dataset.jumpStage = item.stage;
        el.innerHTML = `
          <div class="stage-ans-head">
            <span class="stage-ans-label"></span>
            <span class="stage-ans-badge"></span>
          </div>
          <div class="stage-ans-quote"></div>
        `;
        el.addEventListener('click', () => {
          const st = el.dataset.jumpStage as TrainingStage;
          if (st) {
            this.lossHistory = [];
            this.sendCommand({ type: 'SET_STAGE', stage: st });
          }
        });
        container.appendChild(el);
      });
    }

    // Mutate only class & text in place
    frame.stageAnswers.forEach((item, idx) => {
      const el = container.children[idx] as HTMLElement;
      const isCurrent = item.stage === frame.stage;
      el.classList.toggle('active', isCurrent);
      (el.querySelector('.stage-ans-label') as HTMLElement).textContent =
        `${isCurrent ? '👉 ' : ''}${item.label}`;
      (el.querySelector('.stage-ans-badge') as HTMLElement).textContent = item.badge;
      (el.querySelector('.stage-ans-quote') as HTMLElement).textContent = item.response;
    });
  }

  private renderVramBreakdownInPlace(frame: TelemetryFrame): void {
    const vb = frame.vramBreakdown;
    if (!vb) return;
    const total = Math.max(1, vb.totalMB);
    const wPct = Math.round((vb.weightsMB / total) * 100);
    const gPct = Math.max(1, Math.round((vb.gradientsMB / total) * 100));
    const oPct = Math.max(1, Math.round((vb.optimizerMB / total) * 100));
    const rPct = vb.refModelMB > 0 ? Math.round((vb.refModelMB / total) * 100) : 0;
    const aPct = Math.max(4, 100 - wPct - gPct - oPct - rPct);

    const wEl = document.getElementById('vseg-weights')!;
    const gEl = document.getElementById('vseg-grads')!;
    const oEl = document.getElementById('vseg-opt')!;
    const rEl = document.getElementById('vseg-ref')!;
    const aEl = document.getElementById('vseg-act')!;

    wEl.style.width = `${wPct}%`;
    gEl.style.width = `${gPct}%`;
    oEl.style.width = `${oPct}%`;
    rEl.style.width = `${rPct}%`;
    aEl.style.width = `${aPct}%`;

    const refLegend =
      vb.refModelMB > 0
        ? `<span><b style="color:#7dd3fc">■</b> Bản gốc π_ref: ${vb.refModelMB}MB</span>`
        : '';

    document.getElementById('vram-legend')!.innerHTML = `
      <span><b style="color:#38bdf8">■</b> Trọng số: ${vb.weightsMB}MB</span>
      <span><b style="color:#fb7185">■</b> Grad: ${vb.gradientsMB}MB</span>
      <span><b style="color:#f59e0b">■</b> AdamW (12B): ${vb.optimizerMB}MB</span>
      ${refLegend}
      <span><b style="color:#a855f7">■</b> Kích hoạt: ${vb.activationsMB}MB</span>
    `;
  }

  private drawLossChart(): void {
    const canvas = document.getElementById('loss-chart') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (this.lossHistory.length < 2) return;
    const minL = Math.min(...this.lossHistory) * 0.92;
    const maxL = Math.max(...this.lossHistory) * 1.04 + 0.05;

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    this.lossHistory.forEach((val, idx) => {
      const x = (idx / (this.lossHistory.length - 1)) * (w - 16) + 8;
      const y = h - 8 - ((val - minL) / (maxL - minL)) * (h - 16);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  private renderDynamicPanelInPlace(frame: TelemetryFrame, isSimple: boolean): void {
    const titleEl = document.getElementById('dynamic-panel-title')!;
    const badgeEl = document.getElementById('dynamic-panel-badge')!;
    const bodyEl = document.getElementById('dynamic-panel-body')!;
    const isCollapsed = document.getElementById('body-inspect')?.classList.contains('collapsed');
    const caretHtml = `<span class="caret">${isCollapsed ? '▸' : '▾'}</span>`;

    if (frame.stage === 'rl_dpo' && frame.rl) {
      titleEl.innerHTML = `${caretHtml} ${
        isSimple ? 'RL DPO & Soi Khối 3D' : 'RL DPO Margin & 3D Inspector'
      }`;
      badgeEl.textContent = `KL: ${frame.rl.klDivergence}`;
      const winPct = Math.min(100, Math.max(10, (frame.rl.chosenReward + 1) * 35));
      const losePct = Math.min(100, Math.max(10, (frame.rl.rejectedReward + 2) * 25));
      this.ensureBarRows(bodyEl, 2);
      this.updateBarRow(bodyEl.children[0] as HTMLElement, 'Bài hay (+)', winPct, '#34d399', String(frame.rl.chosenReward));
      this.updateBarRow(bodyEl.children[1] as HTMLElement, 'Bài dở (-)', losePct, '#fb7185', String(frame.rl.rejectedReward));
      return;
    }

    if (frame.stage === 'rl_grpo' && frame.rl) {
      titleEl.innerHTML = `${caretHtml} ${
        isSimple ? 'RL GRPO (4 Bài nháp) & Soi 3D' : 'RL GRPO Group Advantage & 3D Inspector'
      }`;
      badgeEl.textContent = `KL: ${frame.rl.klDivergence}`;
      this.ensureBarRows(bodyEl, frame.rl.groupAdvantages.length);
      frame.rl.groupAdvantages.forEach((adv, idx) => {
        const pct = Math.min(100, Math.max(12, (adv + 1.5) * 32));
        const color = adv >= 0 ? '#34d399' : '#fb7185';
        this.updateBarRow(
          bodyEl.children[idx] as HTMLElement,
          `Bài nháp ${idx + 1}`,
          pct,
          color,
          `${adv >= 0 ? '+' : ''}${adv}`
        );
      });
      return;
    }

    if (frame.architecture === 'moe') {
      const l0 = frame.layers[0];
      titleEl.innerHTML = `${caretHtml} ${
        isSimple ? '4 Bác sĩ Chuyên khoa MoE & Soi 3D' : 'MoE Routed Expert Load & 3D Inspector'
      }`;
      badgeEl.textContent = `Aux Loss: ${l0?.auxLoss ?? 0}`;
      const activeToken = frame.currentTokens[frame.activeTokenIdx];
      const loads = l0?.expertLoads || [];
      this.ensureBarRows(bodyEl, loads.length);
      loads.forEach((load, eIdx) => {
        const isTop = activeToken?.activeExperts.includes(eIdx);
        const pct = Math.min(100, Math.round(load * 240));
        const color = isTop ? '#f59e0b' : '#38bdf8';
        this.updateBarRow(
          bodyEl.children[eIdx] as HTMLElement,
          `Chuyên gia E${eIdx} ${isTop ? '⚡' : ''}`,
          pct,
          color,
          `${Math.round(load * 100)}%`
        );
      });
    } else {
      titleEl.innerHTML = `${caretHtml} ${
        isSimple ? 'Mức độ Vặn núm & Soi Khối 3D' : 'L2 Gradient Norm & 3D Inspector'
      }`;
      badgeEl.textContent = 'GQA 8Q:2KV';
      const slice = frame.layers.slice(0, 4);
      this.ensureBarRows(bodyEl, slice.length);
      slice.forEach((l, idx) => {
        const g = frame.stage === 'lora' ? l.loraAdapterGradNorm : l.baseWeightGradNorm;
        const pct = Math.min(100, Math.round(g * 130));
        const color = frame.stage === 'lora' ? '#e879f9' : '#22d3ee';
        this.updateBarRow(
          bodyEl.children[idx] as HTMLElement,
          `Tầng ${l.layerIdx}`,
          pct,
          color,
          `||g||=${g}`
        );
      });
    }
  }

  private ensureBarRows(container: HTMLElement, count: number): void {
    if (container.children.length === count) return;
    container.innerHTML = Array.from(
      { length: count },
      () => `
        <div class="bar-row">
          <span class="bar-label"></span>
          <div class="bar-track"><div class="bar-fill"></div></div>
          <span class="mono-text bar-val"></span>
        </div>
      `
    ).join('');
  }

  private updateBarRow(
    row: HTMLElement,
    label: string,
    pct: number,
    color: string,
    valText: string
  ): void {
    (row.querySelector('.bar-label') as HTMLElement).textContent = label;
    const fill = row.querySelector('.bar-fill') as HTMLElement;
    fill.style.width = `${pct}%`;
    fill.style.background = color;
    (row.querySelector('.bar-val') as HTMLElement).textContent = valText;
  }

  private renderInspector(meta: InspectableMetadata, frame: TelemetryFrame | null): void {
    document.getElementById('inspect-badge')!.textContent = `TẦNG ${meta.layerIdx}`;
    const layer = frame?.layers[meta.layerIdx];
    const baseGrad = layer?.baseWeightGradNorm ?? 0;
    const loraGrad = layer?.loraAdapterGradNorm ?? 0;
    const desc = this.explanationLevel === 'simple' ? meta.simpleDesc : meta.technicalDesc;

    document.getElementById('inspector-body')!.innerHTML = `
      <div style="font-weight:700;color:#38bdf8;margin-bottom:3px;">${meta.title}</div>
      <div class="mono-text" style="color:#fbbf24;margin-bottom:5px;">Cấu trúc: ${meta.shape}</div>
      <p style="margin-bottom:6px;">${desc}</p>
      <div class="mono-text" style="font-size:10.5px;color:#94a3b8;">
        • Mức vặn Mạng gốc: <b>${baseGrad}</b> ${baseGrad === 0 ? '(🔒 ĐÓNG BĂNG)' : ''}<br/>
        • Mức vặn Ghi chú LoRA: <b>${loraGrad}</b>
      </div>
    `;
  }
}

new MiniMindVisualizerApp();
