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
  activeExperts: number[];    // Indices of Top-K routed experts activated for this token
  expertWeights: number[];    // Gating probabilities for all routed experts
}

export interface LayerTelemetry {
  layerIdx: number;
  attentionEntropy: number;
  kvCacheSavingsRatio: number; // 0.75 for 8 Q heads vs 2 KV heads
  baseWeightGradNorm: number;  // 0 when frozen in LoRA
  loraAdapterGradNorm: number; // > 0 only during LoRA
  expertLoads: number[];       // Normalized load across N routed experts
  auxLoss: number;             // MoE load-balancing auxiliary loss
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
  // Beginner-friendly educational demos
  predictionDemo: NextTokenPredictionDemo;
  comparisonQuestion: string;
  stageAnswers: StageAnswerSample[];
}

export interface ControlCommand {
  type: 'SET_STAGE' | 'SET_ARCH' | 'SET_SPEED' | 'TOGGLE_PAUSE' | 'STEP_ONCE' | 'RESET';
  stage?: TrainingStage;
  architecture?: ArchitectureType;
  speed?: number;
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
