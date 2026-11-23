export type ArchitectureType = 'dense' | 'moe';

export type TrainingStage =
  | 'pretrain'
  | 'sft'
  | 'lora'
  | 'rl_dpo'
  | 'rl_grpo';

export type ExplanationLevel = 'simple' | 'technical';

export interface MiniMindModelConfig {
  architecture: ArchitectureType;
  vocabSize: number;          // MiniMind2: 6400
  hiddenSize: number;         // MiniMind2-Small (Dense): 512 (25.8M), MiniMind2-MoE: 640 (145M)
  numLayers: number;          // Default: 8
  numAttentionHeads: number;  // Default: 8
  numKeyValueHeads: number;   // Default: 2 (GQA 4:1 ratio)
  numRoutedExperts: number;   // Default: 4 routed experts (+ 1 shared)
  numExpertsPerTok: number;   // Default: 2 (Top-2 routing)
  numSharedExperts: number;   // Default: 1
  loraRank: number;           // Default: 8
}

export interface TokenStepInfo {
  tokenId: number;
  positionIdx: number;        // 0-based token position m in the sequence
  text: string;
  role: 'system' | 'user' | 'assistant' | 'raw';
  lossMasked: boolean;        // True for User/System prompt tokens during SFT/LoRA/RL
  ropeAngleRad: number;       // Depends strictly on positionIdx m (m * theta), NOT training step
  activeExperts: number[];    // Top-K routed experts at layer 0 (kept for backward compatibility)
  expertWeights: number[];    // Layer-0 renormalized Top-K weights, 0 for unselected experts
  layerRouting: TokenLayerRouting[]; // MoE only: one routing decision PER LAYER (empty for Dense)
}

/** One router decision for one token in one MoE layer (MiniMind2 MoEGate). */
export interface TokenLayerRouting {
  gateProbs: number[];        // softmax over all routed experts (sums to 1)
  experts: number[];          // Top-K expert indices, highest weight first
  weights: number[];          // Top-K weights renormalized to sum to 1 (norm_topk_prob=True)
}

/** MoE load-balancing experiment state (simulator only). */
export interface MoeBalanceTelemetry {
  auxLossEnabled: boolean;    // false = router trains WITHOUT the load-balancing penalty
  collapse: number;           // 0 = balanced, 1 = fully collapsed onto 2 favourite experts per layer
  effectiveExperts: number;   // exp(entropy) of the load distribution, averaged over layers (4 = perfect, 2 = collapsed)
  auxAlpha: number;           // MiniMind2 aux_loss_alpha = 0.1
  routerFrozen: boolean;      // true in LoRA: the gate is not trained, so balance cannot drift
}

/** Static footprint of one architecture at one stage (used by the Dense vs MoE comparison card). */
export interface ArchFootprint {
  architecture: ArchitectureType;
  totalParamsM: number;
  activeParamsM: number;
  trainableParamsM: number;
  trainablePct: number;
  loraPct: number;
  vramBreakdown: VramBreakdown;
  tokensPerSec: number;
}

export interface LayerTelemetry {
  layerIdx: number;
  attentionEntropy: number;
  kvCacheSavingsRatio: number; // 0.75 for 8 Q heads vs 2 KV heads
  baseWeightGradNorm: number;  // 0 when frozen in LoRA
  loraAdapterGradNorm: number; // > 0 only during LoRA
  expertLoads: number[];       // Fraction of Top-K assignments per routed expert (sums to 1; max 1/K = 50% for Top-2)
  auxLoss: number;             // MiniMind2 seq_aux loss: alpha * sum_i(ce_i * P_i); ~alpha when balanced, ~2*alpha when collapsed
}

export interface RlTelemetry {
  mode: 'dpo' | 'grpo';
  klDivergence: number;
  rewardMargin: number;        // DPO: log_ratio(y_win) - log_ratio(y_lose)
  chosenReward: number;
  rejectedReward: number;
  groupRewards: number[];      // GRPO: rewards of G sampled completions [o_1 ... o_G]
  groupAdvantages: number[];   // GRPO: normalized relative advantages (r_i - mean) / std
}

export interface NextTokenCandidate {
  token: string;
  prob: number;
  isTarget: boolean;
}

export interface NextTokenPredictionDemo {
  promptContext: string;
  targetToken: string;
  targetProb: number;
  tokenSurpriseLoss: number;   // -ln(targetProb) for this specific sample token
  batchAvgLoss: number;        // Corpus/batch average Cross-Entropy loss
  candidates: NextTokenCandidate[];
}

export interface VramBreakdown {
  weightsMB: number;           // 2 bytes/param (bfloat16)
  gradientsMB: number;         // 2 bytes/trainable_param (bfloat16)
  optimizerMB: number;         // 12 bytes/trainable_param (FP32 master + FP32 m + FP32 v)
  refModelMB: number;          // 2 bytes/param for Frozen Reference Model (DPO / GRPO only)
  activationsMB: number;       // Activation buffers & KV rollout cache
  totalMB: number;
}

export interface StageAnswerSample {
  stage: TrainingStage;
  label: string;
  badge: string;
  response: string;
  takeaway: string;
}

export interface TelemetryFrame {
  timestamp: number;
  step: number;
  epoch: number;
  stage: TrainingStage;
  architecture: ArchitectureType;
  source: 'simulator' | 'pytorch_live';
  paused: boolean;
  speed: number;
  cycleSeconds: number;        // Continuous 0..4.8s position in step cycle for 60fps client interpolation
  stepDurationSec: number;     // 4.8s default
  // Core training metrics
  loss: number;                // Batch average loss (realistic scale: ~6.8 -> 2.25 in Pretrain, etc.)
  learningRate: number;
  tokensPerSec: number;
  vramUsedMB: number;
  vramBreakdown: VramBreakdown;
  activeParamsM: number;
  totalParamsM: number;
  trainableParamsM: number;
  trainablePct: number;        // Exact percentage: trainableParamsM / totalParamsM * 100
  // Synchronized 3-phase animation state
  currentTokens: TokenStepInfo[];
  activeTokenIdx: number;
  phase: 'forward' | 'loss_compute' | 'backward';
  phaseProgress: number;       // 0.0 -> 1.0 within current phase
  layers: LayerTelemetry[];
  rl?: RlTelemetry;
  moeBalance?: MoeBalanceTelemetry;
  // Beginner-friendly educational demos
  predictionDemo: NextTokenPredictionDemo;
  comparisonQuestion: string;
  stageAnswers: StageAnswerSample[];
}

export interface ControlCommand {
  type: 'SET_STAGE' | 'SET_ARCH' | 'SET_SPEED' | 'TOGGLE_PAUSE' | 'STEP_ONCE' | 'RESET' | 'SET_AUX_LOSS';
  stage?: TrainingStage;
  architecture?: ArchitectureType;
  speed?: number;
  auxLossEnabled?: boolean;
}

export const DEFAULT_MINIMIND_CONFIG: MiniMindModelConfig = {
  architecture: 'moe',
  vocabSize: 6400,
  hiddenSize: 640,
  numLayers: 8,
  numAttentionHeads: 8,
  numKeyValueHeads: 2,
  numRoutedExperts: 4,
  numExpertsPerTok: 2,
  numSharedExperts: 1,
  loraRank: 8,
};

export * from './simulator.js';
