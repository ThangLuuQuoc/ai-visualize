import {
  ControlCommand,
  DEFAULT_MINIMIND_CONFIG,
  LayerTelemetry,
  MiniMindModelConfig,
  RlTelemetry,
  TelemetryFrame,
  TokenStepInfo,
  TrainingStage,
} from '@minimind-viz/shared';

const PRETRAIN_TOKENS: Array<{ text: string; id: number; role: TokenStepInfo['role'] }> = [
  { text: '<|bos|>', id: 1, role: 'raw' },
  { text: 'MiniMind', id: 412, role: 'raw' },
  { text: 'tối_ưu', id: 1089, role: 'raw' },
  { text: 'vocab', id: 2140, role: 'raw' },
  { text: '6400', id: 6399, role: 'raw' },
  { text: 'với', id: 318, role: 'raw' },
  { text: 'GQA', id: 4821, role: 'raw' },
  { text: 'và', id: 112, role: 'raw' },
  { text: 'FlashAttn', id: 5910, role: 'raw' },
  { text: '<|eos|>', id: 2, role: 'raw' },
];

const CHAT_TOKENS: Array<{ text: string; id: number; role: TokenStepInfo['role'] }> = [
  { text: '<|im_start|>user', id: 3, role: 'user' },
  { text: 'Vì_sao', id: 814, role: 'user' },
  { text: 'MoE', id: 2911, role: 'user' },
  { text: 'train_nhanh?', id: 3402, role: 'user' },
  { text: '<|im_end|>', id: 4, role: 'user' },
  { text: '<|im_start|>assistant', id: 5, role: 'assistant' },
  { text: 'Chỉ_kích_hoạt', id: 1920, role: 'assistant' },
  { text: 'Top-2', id: 4412, role: 'assistant' },
  { text: 'Experts', id: 5108, role: 'assistant' },
  { text: '(64M_active)', id: 6012, role: 'assistant' },
];

export class LocalFallbackSimulator {
  private config: MiniMindModelConfig = { ...DEFAULT_MINIMIND_CONFIG };
  private stage: TrainingStage = 'pretrain';
  private step = 1;
  private subTick = 0;
  private paused = false;

  public applyCommand(cmd: ControlCommand): void {
    if (cmd.type === 'SET_STAGE' && cmd.stage) {
      this.stage = cmd.stage;
      this.step = 1;
      this.subTick = 0;
    } else if (cmd.type === 'SET_ARCH' && cmd.architecture) {
      this.config.architecture = cmd.architecture;
    } else if (cmd.type === 'TOGGLE_PAUSE') {
      this.paused = !this.paused;
    } else if (cmd.type === 'STEP_ONCE') {
      this.paused = true;
      this.subTick += 1;
      if (this.subTick >= 3) {
        this.subTick = 0;
        this.step += 1;
      }
    }
  }

  public isPaused(): boolean {
    return this.paused;
  }

  public tick(): TelemetryFrame {
    if (!this.paused) {
      this.subTick += 1;
      if (this.subTick >= 3) {
        this.subTick = 0;
        this.step += 1;
      }
    }
    return this.buildFrame();
  }

  public buildFrame(): TelemetryFrame {
    const isMoE = this.config.architecture === 'moe';
    const rawPhase = Math.floor(this.subTick) % 3;
    const phase: TelemetryFrame['phase'] =
      rawPhase === 0 ? 'forward' : rawPhase === 1 ? 'loss_compute' : 'backward';

    const totalParamsM = isMoE ? 198.4 : 64.2;
    const activeParamsM = 64.2;
    let trainableParamsM = totalParamsM;
    let vramUsedMB = isMoE ? 2650 : 1480;
    let baseLr = 5e-4;
    let tokensPerSec = isMoE ? 21400 : 25800;

    if (this.stage === 'sft') {
      baseLr = 5e-5;
      vramUsedMB = isMoE ? 2500 : 1420;
    } else if (this.stage === 'lora') {
      trainableParamsM = 1.24;
      baseLr = 1e-4;
      vramUsedMB = isMoE ? 1120 : 690;
      tokensPerSec = isMoE ? 29500 : 34200;
    } else if (this.stage === 'rl_dpo') {
      baseLr = 5e-6;
      vramUsedMB = isMoE ? 3850 : 2240;
      tokensPerSec = 12800;
    } else if (this.stage === 'rl_grpo') {
      baseLr = 2e-6;
      vramUsedMB = isMoE ? 3100 : 1890;
      tokensPerSec = 10900;
    }

    const progress = Math.min(1, this.step / 300);
    const cosineFactor = 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress));
    const learningRate = Number((baseLr * cosineFactor).toExponential(3));

    const initialLoss =
      this.stage === 'pretrain'
        ? 7.85
        : this.stage === 'sft'
          ? 2.65
          : this.stage === 'lora'
            ? 2.15
            : 0.69;
    const targetLoss =
      this.stage === 'pretrain'
        ? 2.15
        : this.stage === 'sft'
          ? 0.92
          : this.stage === 'lora'
            ? 0.84
            : 0.28;
    const noise = (Math.sin(this.step * 0.7) + Math.cos(this.step * 1.3)) * 0.035;
    const loss = Number(
      Math.max(
        targetLoss * 0.85,
        targetLoss + (initialLoss - targetLoss) * Math.exp(-this.step / 65) + noise
      ).toFixed(4)
    );

    const sourceTokens = this.stage === 'pretrain' ? PRETRAIN_TOKENS : CHAT_TOKENS;
    const activeTokenIdx = this.step % sourceTokens.length;
    const numExperts = this.config.numRoutedExperts;

    const currentTokens: TokenStepInfo[] = sourceTokens.map((tok, idx) => {
      const e1 = (tok.id + idx + Math.floor(this.step / 12)) % numExperts;
      const e2 = (e1 + 2 + (tok.id % 3)) % numExperts;
      const weights = Array.from({ length: numExperts }, (_, eIdx) => {
        if (!isMoE) return 0;
        if (eIdx === e1) return 0.58;
        if (eIdx === e2) return 0.42;
        return 0.0;
      });
      const isPromptMasked =
        this.stage !== 'pretrain' && (tok.role === 'user' || tok.role === 'system');

      return {
        tokenId: tok.id,
        text: tok.text,
        role: tok.role,
        lossMasked: isPromptMasked,
        ropeAngleRad: Number(((idx * 0.628) % (Math.PI * 2)).toFixed(3)),
        activeExperts: isMoE ? [e1, e2] : [],
        expertWeights: weights,
      };
    });

    const layers: LayerTelemetry[] = Array.from({ length: this.config.numLayers }, (_, layerIdx) => {
      const depthFactor = 1 - layerIdx / (this.config.numLayers * 1.6);
      const isLora = this.stage === 'lora';
      const baseWeightGradNorm = isLora
        ? 0
        : Number((0.42 * depthFactor * (0.85 + 0.3 * Math.sin(this.step * 0.3 + layerIdx))).toFixed(3));
      const loraAdapterGradNorm = isLora
        ? Number((0.68 * depthFactor * (0.85 + 0.3 * Math.cos(this.step * 0.4 + layerIdx))).toFixed(3))
        : 0;

      const imbalance = Math.exp(-this.step / 80) * 0.16;
      const rawLoads = Array.from({ length: numExperts }, (__, eIdx) => {
        const wave = Math.sin(this.step * 0.15 + layerIdx + eIdx * 1.4) * imbalance;
        return Math.max(0.05, 1 / numExperts + wave);
      });
      const sumLoads = rawLoads.reduce((a, b) => a + b, 0);
      const expertLoads = rawLoads.map((v) => Number((v / sumLoads).toFixed(3)));

      return {
        layerIdx,
        attentionEntropy: Number((1.4 + 0.4 * Math.sin(layerIdx + this.step * 0.08)).toFixed(3)),
        kvCacheSavingsRatio: 0.75,
        baseWeightGradNorm,
        loraAdapterGradNorm,
        expertLoads,
        auxLoss: isMoE ? Number((0.025 * Math.exp(-this.step / 90) + 0.004).toFixed(4)) : 0,
      };
    });

    let rl: RlTelemetry | undefined;
    if (this.stage === 'rl_dpo' || this.stage === 'rl_grpo') {
      const rlProgress = 1 - Math.exp(-this.step / 55);
      const chosenReward = Number((0.45 + 1.35 * rlProgress + noise * 0.4).toFixed(3));
      const rejectedReward = Number((-0.25 - 1.1 * rlProgress - noise * 0.4).toFixed(3));
      const groupRewards = [
        Number((0.88 + 0.4 * rlProgress + Math.sin(this.step) * 0.08).toFixed(2)),
        Number((0.52 + 0.2 * rlProgress + Math.cos(this.step) * 0.07).toFixed(2)),
        Number((0.15 - 0.1 * rlProgress).toFixed(2)),
        Number((-0.45 - 0.3 * rlProgress).toFixed(2)),
      ];
      const meanR = groupRewards.reduce((a, b) => a + b, 0) / groupRewards.length;
      const stdR =
        Math.sqrt(
          groupRewards.reduce((acc, r) => acc + Math.pow(r - meanR, 2), 0) / groupRewards.length
        ) || 1;
      const groupAdvantages = groupRewards.map((r) => Number(((r - meanR) / stdR).toFixed(2)));

      rl = {
        mode: this.stage === 'rl_dpo' ? 'dpo' : 'grpo',
        klDivergence: Number((0.018 + 0.045 * rlProgress).toFixed(4)),
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
      epoch: Math.floor(this.step / 100) + 1,
      stage: this.stage,
      architecture: this.config.architecture,
      source: 'simulator',
      loss,
      learningRate,
      tokensPerSec,
      vramUsedMB,
      activeParamsM,
      totalParamsM,
      trainableParamsM,
      currentTokens,
      activeTokenIdx,
      phase,
      layers,
      rl,
    };
  }
}
