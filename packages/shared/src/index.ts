export type ArchitectureType = 'dense' | 'moe';

export type TrainingStage =
  | 'pretrain'
  | 'sft'
  | 'lora'
  | 'rl_dpo'
  | 'rl_grpo';

export interface MiniMindModelConfig {
  architecture: ArchitectureType;
  vocabSize: number;          // Default: 6400
  hiddenSize: number;         // Default: 512 or 768
  numLayers: number;          // Default: 8
  numAttentionHeads: number;  // Default: 8
  numKeyValueHeads: number;   // Default: 2 (GQA 4:1 ratio)
  // MoE specific
  numRoutedExperts: number;   // Default: 4 or 8
  numExpertsPerTok: number;   // Default: 2 (Top-2 routing)
  numSharedExperts: number;   // Default: 1
  // LoRA specific
  loraRank: number;           // Default: 8
}

export interface TokenStepInfo {
  tokenId: number;
  text: string;
  role: 'system' | 'user' | 'assistant' | 'raw';
  lossMasked: boolean;        // True for User/System prompt tokens during SFT/LoRA/RL
  ropeAngleRad: number;       // Rotary position phase angle
  activeExperts: number[];    // Indices of Top-K routed experts activated for this token
  expertWeights: number[];    // Gating probabilities for all routed experts
}

export interface LayerTelemetry {
  layerIdx: number;
  attentionEntropy: number;
  kvCacheSavingsRatio: number; // e.g. 0.75 for 8 Q heads vs 2 KV heads
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

export interface TelemetryFrame {
  timestamp: number;
  step: number;
  epoch: number;
  stage: TrainingStage;
  architecture: ArchitectureType;
  source: 'simulator' | 'pytorch_live';
  // Core training metrics
  loss: number;
  learningRate: number;
  tokensPerSec: number;
  vramUsedMB: number;
  activeParamsM: number;
  totalParamsM: number;
  trainableParamsM: number;
  // Detailed 3D visualization states
  currentTokens: TokenStepInfo[];
  activeTokenIdx: number;
  phase: 'forward' | 'loss_compute' | 'backward';
  layers: LayerTelemetry[];
  rl?: RlTelemetry;
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
  hiddenSize: 768,
  numLayers: 8,
  numAttentionHeads: 8,
  numKeyValueHeads: 2,
  numRoutedExperts: 6,
  numExpertsPerTok: 2,
  numSharedExperts: 1,
  loraRank: 8,
};
