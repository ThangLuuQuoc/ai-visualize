import {
  ArchFootprint,
  ArchitectureType,
  ControlCommand,
  DEFAULT_MINIMIND_CONFIG,
  LayerTelemetry,
  MiniMindModelConfig,
  NextTokenPredictionDemo,
  RlTelemetry,
  StageAnswerSample,
  MoeBalanceTelemetry,
  TelemetryFrame,
  TokenLayerRouting,
  TokenStepInfo,
  TrainingStage,
  VramBreakdown,
} from './index.js';

const PRETRAIN_TOKENS: Array<{ text: string; id: number; role: TokenStepInfo['role'] }> = [
  { text: '<|bos|>', id: 1, role: 'raw' },
  { text: 'Hà_Nội', id: 412, role: 'raw' },
  { text: 'là', id: 108, role: 'raw' },
  { text: 'thủ_đô', id: 2140, role: 'raw' },
  { text: 'của', id: 318, role: 'raw' },
  { text: 'Việt_Nam', id: 4821, role: 'raw' },
  { text: 'với', id: 112, role: 'raw' },
  { text: 'lịch_sử', id: 5910, role: 'raw' },
  { text: 'lâu_đời', id: 6104, role: 'raw' },
  { text: '<|eos|>', id: 2, role: 'raw' },
];

const CHAT_TOKENS: Array<{ text: string; id: number; role: TokenStepInfo['role'] }> = [
  { text: '<|user|>', id: 3, role: 'user' },
  { text: 'Thủ_đô', id: 814, role: 'user' },
  { text: 'Việt_Nam', id: 2911, role: 'user' },
  { text: 'là_gì?', id: 3402, role: 'user' },
  { text: '<|assistant|>', id: 5, role: 'assistant' },
  { text: 'Thủ_đô', id: 814, role: 'assistant' },
  { text: 'của', id: 318, role: 'assistant' },
  { text: 'Việt_Nam', id: 2911, role: 'assistant' },
  { text: 'là', id: 108, role: 'assistant' },
  { text: 'Hà_Nội.', id: 412, role: 'assistant' },
];

const COMPARISON_QUESTION = 'Câu hỏi thử nghiệm: "Làm sao để học lập trình hiệu quả cho người mới?"';

function buildStageAnswers(trainablePct: number): StageAnswerSample[] {
  return [
    {
      stage: 'pretrain',
      label: '1. Sau Pretrain (Đọc sách thô)',
      badge: 'Chỉ biết viết tiếp câu',
      response:
        '«Làm sao để học lập trình hiệu quả cho người mới? Câu hỏi này được đăng ngày 12/05 trên diễn đàn cùng 45 bình luận khác nhau về giáo trình...»',
      takeaway:
        'Mô hình đã thuộc ngữ pháp và từ vựng, nhưng chỉ có quán tính viết nối tiếp văn bản chứ chưa hiểu người dùng đang cần một lời giải đáp.',
    },
    {
      stage: 'sft',
      label: '2. Sau Full SFT (Luyện khuôn Hỏi–Đáp)',
      badge: 'Trả lời đúng trọng tâm',
      response:
        '«Bạn nên chọn một ngôn ngữ dễ đọc như Python, học vững biến, vòng lặp, hàm và tự tay làm các dự án nhỏ để rèn tư duy.»',
      takeaway:
        'Nhờ chỉ tính Loss ở phần Assistant (nhưng vẫn truyền đạo hàm qua ngữ cảnh câu hỏi User), mô hình hình thành phản xạ trả lời trực tiếp.',
    },
    {
      stage: 'lora',
      label: '3. Sau LoRA (Dán thêm ghi chú chuyên biệt)',
      badge: `Chỉ chỉnh ${trainablePct}% tham số`,
      response:
        '«[Lộ trình 4 tuần]: Tuần 1-2 học cú pháp Python; Tuần 3 viết 2 script tự động hóa; Tuần 4 đọc hiểu code trên GitHub và sửa lỗi.»',
      takeaway:
        'Đóng băng 100% mạng gốc và chỉ cập nhật nhánh phụ A × B (Rank-8), giúp tiết kiệm mạnh bộ nhớ AdamW Optimizer mà vẫn dạy được khuôn mẫu mới.',
    },
    {
      stage: 'rl_dpo',
      label: '4a. Sau RL - DPO (Thầy giáo so sánh 2 bài)',
      badge: 'Thân thiện, rõ ràng & an toàn',
      response:
        '«Chào bạn! Để bắt đầu nhẹ nhàng, bạn hãy: 1) Chọn 1 ngôn ngữ nền tảng (Python), 2) Code thực hành 45 phút mỗi ngày thay vì chỉ xem video, 3) Tự làm 1 dự án nhỏ bạn thích. Chúc bạn kiên trì!»',
      takeaway:
        'Tối ưu trực tiếp độ chênh lệch giữa bài trả lời tốt (y_win) và bài cụt lủn (y_lose) có neo KL với bản gốc, giúp câu trả lời mạch lạc và lịch sự hơn.',
    },
    {
      stage: 'rl_grpo',
      label: '4b. Sau RL - GRPO (Tự chấm điểm nhóm 4 bài)',
      badge: 'Lập luận chặt chẽ, không tốn Critic',
      response:
        '«Gợi ý 3 bước cốt lõi: (Bước 1 - Nền tảng) Hiểu logic điều kiện & cấu trúc dữ liệu; (Bước 2 - Thực hành) Tự giải 30 bài toán nhỏ; (Bước 3 - Phản hồi) Dùng debugger để hiểu tận gốc mỗi khi code báo lỗi.»',
      takeaway:
        'Tự sinh 4 phương án trả lời rồi dùng điểm trung bình nhóm làm mốc thưởng/phạt, loại bỏ hoàn toàn mô hình Giám khảo (Critic) của PPO.',
    },
  ];
}

/** Deterministic hash -> [0, 1). Same inputs always give the same "router opinion". */
function hash01(a: number, b: number, c: number, d: number): number {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b);
  h = Math.imul(h ^ (b + 0x632be5ab), 0xc2b2ae35);
  h = Math.imul(h ^ (c + 0x27d4eb2f), 0x165667b1);
  h = Math.imul(h ^ (d + 0x7f4a7c15), 0x85ebca6b);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

/** MiniMind2 aux_loss_alpha (model_minimind.py, MiniMindConfig default). */
const AUX_ALPHA = 0.1;
/** Virtual batch used to estimate per-layer expert load each step (the visible sentence is only 10 tokens). */
const LOAD_BATCH_TOKENS = 64;

/**
 * The 2 experts each layer drifts towards when the router trains without aux loss.
 * In reality which experts "win" is an accident of initialisation; here it is fixed per layer.
 */
export function favouredExpertPair(layerIdx: number, numExperts = 4): [number, number] {
  const a = layerIdx % numExperts;
  const b = (a + 1 + (layerIdx % (numExperts - 1))) % numExperts;
  return [a, b];
}

export class MiniMindSimulator {
  private config: MiniMindModelConfig = { ...DEFAULT_MINIMIND_CONFIG };
  private stage: TrainingStage = 'pretrain';
  private step = 1;
  private cycleSeconds = 0;
  private paused = false;
  private speed = 1.0;
  // MoE load-balancing experiment: start slightly unbalanced so aux loss is seen pulling it back.
  private auxLossEnabled = true;
  private static readonly INITIAL_COLLAPSE = 0.3;
  private collapse = MiniMindSimulator.INITIAL_COLLAPSE;

  public readonly STEP_DURATION_SEC = 4.8;

  /**
   * Params, VRAM and throughput for one architecture at one stage.
   * Shared by buildFrame() and the Dense vs MoE comparison card so both always agree.
   */
  public static computeFootprint(
    architecture: ArchitectureType,
    stage: TrainingStage,
    loraRank = DEFAULT_MINIMIND_CONFIG.loraRank,
    numLayers = DEFAULT_MINIMIND_CONFIG.numLayers
  ): ArchFootprint {
    const isMoE = architecture === 'moe';
    // Authentic MiniMind2 repository specifications:
    // - MiniMind2-Small (Dense): hidden_size=512, num_layers=8, vocab=6400 -> 25.8M parameters
    // - MiniMind2-MoE: hidden_size=640, num_layers=8, 4 Routed + 1 Shared (Top-2) -> 145.0M total / 45.2M active
    const totalParamsM = isMoE ? 145.0 : 25.8;
    const activeParamsM = isMoE ? 45.2 : 25.8;
    // MiniMind's apply_lora() only wraps *square* nn.Linear layers (in_features == out_features).
    // With GQA (8Q : 2KV), k_proj/v_proj are not square, nor are FFN/MoE/lm_head, so per layer
    // only q_proj and o_proj (d -> d) get LoRA. Each adds A[d×r] + B[r×d] = 2·r·d params.
    // Total = layers × 2 matrices × 2·r·d (r = 8):
    //   Dense (d=512): 8 × 2 × 2 × 8 × 512 = 131,072 ≈ 0.13M (0.51% of 25.8M)
    //   MoE   (d=640): 8 × 2 × 2 × 8 × 640 = 163,840 ≈ 0.16M (0.11% of 145.0M)
    const loraDim = isMoE ? 640 : 512;
    const loraParamsM = Number(((numLayers * 2 * 2 * loraRank * loraDim) / 1e6).toFixed(2));
    const loraPct = Number(((loraParamsM / totalParamsM) * 100).toFixed(2));
    const trainableParamsM = stage === 'lora' ? loraParamsM : totalParamsM;
    const trainablePct = Number(((trainableParamsM / totalParamsM) * 100).toFixed(2));

    // Exact Mixed-Precision (bfloat16 weights/grads + FP32 AdamW master/m/v) VRAM breakdown:
    // - Model Weights (bf16): 2 bytes/param
    // - Gradients (bf16): 2 bytes/trainable_param
    // - AdamW Optimizer States (FP32 master copy 4B + FP32 momentum m 4B + FP32 variance v 4B): 12 bytes/trainable_param
    // - Frozen Reference Model (bf16, only in DPO & GRPO): 2 bytes/param (separated into its own segment!)
    const weightsMB = Math.round(totalParamsM * 2);
    const gradientsMB = Math.max(2, Math.round(trainableParamsM * 2));
    const optimizerMB = Math.max(8, Math.round(trainableParamsM * 12));
    const isRl = stage === 'rl_dpo' || stage === 'rl_grpo';
    const refModelMB = isRl ? weightsMB : 0;

    let activationsMB = isMoE ? 540 : 320;
    let tokensPerSec = isMoE ? 18400 : 26500;
    if (stage === 'lora') {
      activationsMB = isMoE ? 420 : 250;
      // Modest ~13% throughput increase since Forward and Activation-Backward still traverse the full network
      tokensPerSec = isMoE ? 20800 : 29800;
    } else if (stage === 'rl_dpo') {
      activationsMB = isMoE ? 640 : 410;
      tokensPerSec = isMoE ? 11200 : 15800;
    } else if (stage === 'rl_grpo') {
      // Group sampling G=4 increases activation & KV cache footprint, but avoids a Critic model
      activationsMB = isMoE ? 820 : 530;
      tokensPerSec = isMoE ? 8900 : 12400;
    }

    const totalMB = weightsMB + gradientsMB + optimizerMB + refModelMB + activationsMB;
    return {
      architecture,
      totalParamsM,
      activeParamsM,
      trainableParamsM,
      trainablePct,
      loraPct,
      vramBreakdown: { weightsMB, gradientsMB, optimizerMB, refModelMB, activationsMB, totalMB },
      tokensPerSec,
    };
  }

  public getConfig(): MiniMindModelConfig {
    return this.config;
  }

  public isPaused(): boolean {
    return this.paused;
  }

  public getSpeed(): number {
    return this.speed;
  }

  /**
   * Synchronize stage/architecture/step from an incoming live PyTorch frame
   * WITHOUT resetting step to 1 or zeroing cycleSeconds!
   */
  public syncFromLivePyTorch(incoming: {
    stage?: TrainingStage;
    architecture?: ArchitectureType;
    step?: number;
  }): void {
    if (incoming.stage && incoming.stage !== this.stage) {
      this.stage = incoming.stage;
    }
    if (incoming.architecture && incoming.architecture !== this.config.architecture) {
      this.config.architecture = incoming.architecture;
      this.config.hiddenSize = incoming.architecture === 'moe' ? 640 : 512;
    }
    if (typeof incoming.step === 'number' && incoming.step > 0) {
      this.step = incoming.step;
    }
  }

  public applyCommand(cmd: ControlCommand): void {
    switch (cmd.type) {
      case 'SET_STAGE':
        // Only reset step & cycleSeconds if the user actually switches to a different stage
        if (cmd.stage && cmd.stage !== this.stage) {
          this.stage = cmd.stage;
          this.step = 1;
          this.cycleSeconds = 0;
        }
        break;
      case 'SET_ARCH':
        if (cmd.architecture && cmd.architecture !== this.config.architecture) {
          this.config.architecture = cmd.architecture;
          this.config.hiddenSize = cmd.architecture === 'moe' ? 640 : 512;
        }
        break;
      case 'SET_SPEED':
        if (typeof cmd.speed === 'number' && !Number.isNaN(cmd.speed)) {
          this.speed = Math.max(0.25, Math.min(3.0, cmd.speed));
        }
        break;
      case 'TOGGLE_PAUSE':
        this.paused = !this.paused;
        break;
      case 'STEP_ONCE':
        this.paused = true;
        this.cycleSeconds += this.STEP_DURATION_SEC / 3;
        if (this.cycleSeconds >= this.STEP_DURATION_SEC) {
          this.cycleSeconds = 0;
          this.advanceStep();
        }
        break;
      case 'RESET':
        this.step = 1;
        this.cycleSeconds = 0;
        this.collapse = MiniMindSimulator.INITIAL_COLLAPSE;
        break;
      case 'SET_AUX_LOSS':
        if (typeof cmd.auxLossEnabled === 'boolean') {
          this.auxLossEnabled = cmd.auxLossEnabled;
        }
        break;
    }
  }

  /**
   * One optimizer step. Also evolves MoE load balance:
   * - Without aux loss the router drifts "rich get richer": experts picked more often get more
   *   gradient, become better, and get picked even more. Logistic growth => slow start, then fast.
   * - With aux loss the penalty pulls the load back towards uniform.
   * - In LoRA the gate is frozen (apply_lora only touches q_proj/o_proj), so nothing changes.
   */
  private advanceStep(): void {
    this.step += 1;
    if (this.config.architecture !== 'moe' || this.stage === 'lora') return;
    if (this.auxLossEnabled) {
      this.collapse *= 0.82;
    } else {
      this.collapse += 0.2 * (this.collapse + 0.1) * (1 - this.collapse);
    }
    this.collapse = Math.min(1, Math.max(0, this.collapse));
  }

  /**
   * MiniMind2 MoEGate for one token in one layer:
   * scores = softmax(gate(x)) -> topk(k=2) -> renormalize the 2 weights (norm_topk_prob=True).
   * gate(x) is faked with a deterministic hash of (token, position, layer, expert) so it is stable
   * across frames, differs per layer, and is biased towards the favoured pair as `collapse` grows.
   */
  private routeToken(seedA: number, seedB: number, layerIdx: number): TokenLayerRouting {
    const n = this.config.numRoutedExperts;
    const k = this.config.numExpertsPerTok;
    const fav = favouredExpertPair(layerIdx, n);
    const logits = Array.from({ length: n }, (_, e) => {
      const base = (hash01(seedA, seedB, layerIdx, e) * 2 - 1) * 1.6;
      return base + (fav.includes(e) ? this.collapse * 4.5 : 0);
    });
    const maxL = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);
    const gateProbs = exps.map((v) => v / sum);
    const experts = gateProbs
      .map((p, e) => ({ p, e }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k)
      .map((x) => x.e);
    const topSum = experts.reduce((acc, e) => acc + gateProbs[e], 0);
    const w0 = Number((gateProbs[experts[0]] / topSum).toFixed(2));
    const weights =
      k === 2
        ? [w0, Number((1 - w0).toFixed(2))]
        : experts.map((e) => Number((gateProbs[e] / topSum).toFixed(2)));
    return {
      gateProbs: gateProbs.map((p) => Number(p.toFixed(3))),
      experts,
      weights,
    };
  }

  /**
   * Per-layer expert load over a virtual batch, plus MiniMind2's seq_aux loss:
   *   ce_i = count_i / (T·k / N)   (1.0 when perfectly balanced)
   *   P_i  = mean softmax score of expert i
   *   aux  = alpha · Σ ce_i · P_i  (≈ alpha balanced, ≈ 2·alpha when 2 of 4 experts take everything)
   */
  private computeLayerLoad(layerIdx: number): { loads: number[]; auxLoss: number } {
    const n = this.config.numRoutedExperts;
    const k = this.config.numExpertsPerTok;
    const counts = new Array<number>(n).fill(0);
    const probSums = new Array<number>(n).fill(0);
    for (let t = 0; t < LOAD_BATCH_TOKENS; t++) {
      const r = this.routeToken(10_000 + this.step * 131 + t, t, layerIdx);
      r.experts.forEach((e) => (counts[e] += 1));
      r.gateProbs.forEach((p, e) => (probSums[e] += p));
    }
    const totalAssign = LOAD_BATCH_TOKENS * k;
    const loads = counts.map((c) => Number((c / totalAssign).toFixed(3)));
    const auxLoss =
      counts.reduce(
        (acc, c, e) => acc + (c / (totalAssign / n)) * (probSums[e] / LOAD_BATCH_TOKENS),
        0
      ) * AUX_ALPHA;
    return { loads, auxLoss: Number(auxLoss.toFixed(4)) };
  }

  public tick(dtSeconds = 0.12): TelemetryFrame {
    if (!this.paused) {
      this.cycleSeconds += dtSeconds * this.speed;
      while (this.cycleSeconds >= this.STEP_DURATION_SEC) {
        this.cycleSeconds -= this.STEP_DURATION_SEC;
        this.advanceStep();
      }
    }
    return this.buildFrame();
  }

  public static computePhaseFromCycle(
    cycleSeconds: number,
    stepDurationSec = 4.8
  ): { phase: TelemetryFrame['phase']; phaseProgress: number } {
    const t = ((cycleSeconds % stepDurationSec) + stepDurationSec) % stepDurationSec;
    if (t < 1.8) {
      return { phase: 'forward', phaseProgress: Number((t / 1.8).toFixed(3)) };
    }
    if (t < 3.0) {
      return { phase: 'loss_compute', phaseProgress: Number(((t - 1.8) / 1.2).toFixed(3)) };
    }
    return {
      phase: 'backward',
      phaseProgress: Number(Math.min(1, (t - 3.0) / 1.8).toFixed(3)),
    };
  }

  public buildFrame(): TelemetryFrame {
    const isMoE = this.config.architecture === 'moe';
    const { phase, phaseProgress } = MiniMindSimulator.computePhaseFromCycle(
      this.cycleSeconds,
      this.STEP_DURATION_SEC
    );

    const fp = MiniMindSimulator.computeFootprint(
      this.config.architecture,
      this.stage,
      this.config.loraRank,
      this.config.numLayers
    );
    const { totalParamsM, activeParamsM, trainableParamsM, trainablePct, loraPct, vramBreakdown, tokensPerSec } =
      fp;
    const vramUsedMB = vramBreakdown.totalMB;
    const isRl = this.stage === 'rl_dpo' || this.stage === 'rl_grpo';
    const baseLr =
      this.stage === 'sft'
        ? 5e-5
        : this.stage === 'lora'
          ? 1e-4
          : this.stage === 'rl_dpo'
            ? 5e-6
            : this.stage === 'rl_grpo'
              ? 2e-6
              : 5e-4;

    const progress = Math.min(1, (this.step - 1) / 35);
    const cosineFactor = 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress));
    const learningRate = Number((baseLr * cosineFactor).toExponential(3));

    const predictionDemo = this.buildNextTokenDemo(progress);
    const loss = predictionDemo.batchAvgLoss;

    const sourceTokens = this.stage === 'pretrain' ? PRETRAIN_TOKENS : CHAT_TOKENS;
    const activeTokenIdx = Math.min(
      sourceTokens.length - 1,
      Math.floor(phaseProgress * sourceTokens.length)
    );
    const numExperts = this.config.numRoutedExperts;

    const currentTokens: TokenStepInfo[] = sourceTokens.map((tok, idx) => {
      const layerRouting = isMoE
        ? Array.from({ length: this.config.numLayers }, (_, l) => this.routeToken(tok.id, idx, l))
        : [];
      const l0 = layerRouting[0];
      const weights = Array.from({ length: numExperts }, (_, eIdx) => {
        if (!l0) return 0;
        const pos = l0.experts.indexOf(eIdx);
        return pos >= 0 ? l0.weights[pos] : 0;
      });
      const isPromptMasked =
        this.stage !== 'pretrain' && (tok.role === 'user' || tok.role === 'system');

      // RoPE rotation angle depends strictly on the token's position index m (m * theta_base), never on training step!
      const ropeAngleRad = Number(((idx * 0.62) % (Math.PI * 2)).toFixed(2));

      return {
        tokenId: tok.id,
        positionIdx: idx,
        text: tok.text,
        role: tok.role,
        lossMasked: isPromptMasked,
        ropeAngleRad,
        activeExperts: l0 ? [...l0.experts] : [],
        expertWeights: weights,
        layerRouting,
      };
    });

    const layers: LayerTelemetry[] = Array.from({ length: this.config.numLayers }, (_, layerIdx) => {
      const depthFactor = 1 - layerIdx / (this.config.numLayers * 1.6);
      const isLora = this.stage === 'lora';
      const baseWeightGradNorm = isLora
        ? 0
        : Number((0.42 * depthFactor * (0.9 + 0.2 * Math.sin(this.step * 0.4 + layerIdx))).toFixed(3));
      const loraAdapterGradNorm = isLora
        ? Number((0.65 * depthFactor * (0.9 + 0.2 * Math.cos(this.step * 0.4 + layerIdx))).toFixed(3))
        : 0;

      const { loads: expertLoads, auxLoss } = isMoE
        ? this.computeLayerLoad(layerIdx)
        : { loads: [] as number[], auxLoss: 0 };

      return {
        layerIdx,
        attentionEntropy: Number((1.35 + 0.3 * Math.sin(layerIdx + this.step * 0.15)).toFixed(3)),
        kvCacheSavingsRatio: 0.75,
        baseWeightGradNorm,
        loraAdapterGradNorm,
        expertLoads,
        auxLoss,
      };
    });

    let moeBalance: MoeBalanceTelemetry | undefined;
    if (isMoE) {
      const effPerLayer = layers.map((l) =>
        Math.exp(-l.expertLoads.reduce((acc, p) => (p > 0 ? acc + p * Math.log(p) : acc), 0))
      );
      moeBalance = {
        auxLossEnabled: this.auxLossEnabled,
        collapse: Number(this.collapse.toFixed(3)),
        effectiveExperts: Number(
          (effPerLayer.reduce((a, b) => a + b, 0) / Math.max(1, effPerLayer.length)).toFixed(2)
        ),
        auxAlpha: AUX_ALPHA,
        routerFrozen: this.stage === 'lora',
      };
    }

    let rl: RlTelemetry | undefined;
    if (isRl) {
      const rlProgress = 1 - Math.exp(-this.step / 15);
      const chosenReward = Number((0.45 + 1.35 * rlProgress).toFixed(3));
      const rejectedReward = Number((-0.25 - 1.05 * rlProgress).toFixed(3));
      const groupRewards = [
        Number((0.92 + 0.35 * rlProgress).toFixed(2)),
        Number((0.48 + 0.18 * rlProgress).toFixed(2)),
        Number((0.12 - 0.1 * rlProgress).toFixed(2)),
        Number((-0.48 - 0.28 * rlProgress).toFixed(2)),
      ];
      const meanR = groupRewards.reduce((a, b) => a + b, 0) / groupRewards.length;
      const stdR =
        Math.sqrt(
          groupRewards.reduce((acc, r) => acc + Math.pow(r - meanR, 2), 0) / groupRewards.length
        ) || 1;
      const groupAdvantages = groupRewards.map((r) => Number(((r - meanR) / stdR).toFixed(2)));

      rl = {
        mode: this.stage === 'rl_dpo' ? 'dpo' : 'grpo',
        klDivergence: Number((0.015 + 0.038 * rlProgress).toFixed(4)),
        rewardMargin: Number((chosenReward - rejectedReward).toFixed(3)),
        chosenReward,
        rejectedReward,
        groupRewards,
        groupAdvantages,
      };
    }

    return {
      timestamp: Date.now(),
      step: this.step,
      epoch: Math.floor((this.step - 1) / 25) + 1,
      stage: this.stage,
      architecture: this.config.architecture,
      source: 'simulator',
      paused: this.paused,
      speed: this.speed,
      cycleSeconds: Number(this.cycleSeconds.toFixed(3)),
      stepDurationSec: this.STEP_DURATION_SEC,
      loss,
      learningRate,
      tokensPerSec,
      vramUsedMB,
      vramBreakdown,
      activeParamsM,
      totalParamsM,
      trainableParamsM,
      trainablePct,
      currentTokens,
      activeTokenIdx,
      phase,
      phaseProgress,
      layers,
      rl,
      moeBalance,
      predictionDemo,
      comparisonQuestion: COMPARISON_QUESTION,
      stageAnswers: buildStageAnswers(loraPct),
    };
  }

  /**
   * Builds both:
   * 1) The single-token prediction probability demo (monotonically inherited across stages so the model never "forgets" what it learned in Pretrain!)
   * 2) The realistic corpus/batch average Loss (Pretrain: 6.85 -> 2.24, SFT: 2.24 -> 1.12, LoRA: 1.48 -> 0.88, RL: 0.69 -> 0.31)
   */
  private buildNextTokenDemo(progress: number): NextTokenPredictionDemo {
    // Monotonic probability progression: each stage starts exactly where its parent ended.
    // Pretrain: 6% -> 72%
    // SFT (starts from the Pretrain checkpoint): 72% -> 88%
    // LoRA and DPO / GRPO both branch from the SFT checkpoint, so both start at 88%:
    //   LoRA: 88% -> 94%   |   DPO / GRPO: 88% -> 97%
    const [minP, maxP] =
      this.stage === 'pretrain'
        ? [0.06, 0.72]
        : this.stage === 'sft'
          ? [0.72, 0.88]
          : this.stage === 'lora'
            ? [0.88, 0.94]
            : [0.88, 0.97];

    const curve = 1 - Math.exp(-progress * 3.1);
    const targetProb = Number((minP + (maxP - minP) * curve).toFixed(3));
    const rem = Math.max(0.01, 1 - targetProb);

    const candidates = [
      { token: 'Hà_Nội', prob: targetProb, isTarget: true },
      { token: 'TP.HCM', prob: Number((rem * 0.42).toFixed(3)), isTarget: false },
      { token: 'Đà_Nẵng', prob: Number((rem * 0.26).toFixed(3)), isTarget: false },
      { token: 'một', prob: Number((rem * 0.19).toFixed(3)), isTarget: false },
      { token: 'những', prob: Number((rem * 0.13).toFixed(3)), isTarget: false },
    ].sort((a, b) => b.prob - a.prob);

    const tokenSurpriseLoss = Number((-Math.log(Math.max(0.001, targetProb))).toFixed(3));

    // Realistic batch-wide average loss across thousands of diverse tokens in the dataset
    const [startBatchLoss, endBatchLoss] =
      this.stage === 'pretrain'
        ? [6.85, 2.24]
        : this.stage === 'sft'
          ? [2.24, 1.12]
          : this.stage === 'lora'
            ? [1.48, 0.88]
            : [0.69, 0.31];
    const batchAvgLoss = Number(
      (endBatchLoss + (startBatchLoss - endBatchLoss) * Math.exp(-progress * 2.8)).toFixed(4)
    );

    return {
      promptContext:
        this.stage === 'pretrain'
          ? 'Câu mẫu trong sách: "Việt Nam là quốc gia Đông Nam Á có thủ đô là [ ___ ]"'
          : 'Hỏi–Đáp: "User: Thủ đô của Việt Nam là gì? → Assistant: Thủ đô của Việt Nam là [ ___ ]"',
      targetToken: 'Hà_Nội',
      targetProb,
      tokenSurpriseLoss,
      batchAvgLoss,
      candidates,
    };
  }
}
