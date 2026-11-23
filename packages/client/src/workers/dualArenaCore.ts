/**
 * Dual Live Training Arena Core (Pure TypeScript).
 *
 * Trains 3 real Language Models simultaneously on the exact same shuffled epoch batches
 * of 64 Vietnamese sentences (880 token samples/epoch):
 *   1. Dense nhỏ (H=18): 18 active / 18 total FFN neurons (tanh).
 *   2. MoE (4×7 Top-2 + 4 Shared): 18 active / 32 total FFN neurons (tanh, norm_topk_prob, SHARED_SCALE=1).
 *   3. Dense to (H=32): 32 active / 32 total FFN neurons (tanh, matching MoE's total FFN width).
 *
 * All 3 models share identical initial Embedding (E) and LM Head (Wout, bout) weights from the same seed,
 * use causal context pooling `x = 0.4·E[cur] + 0.6·mean(E[prev])` (no attention, no residual skip),
 * and save full `DualEpochSnapshot` checkpoints after every complete epoch pass.
 */

import {
  CORPUS,
  DOMAIN_ORDER,
  DomainId,
  ExpertEmergenceSnapshot,
} from './moeEmergenceCore';

export interface ProbeTokenRef {
  id: DomainId;
  word: string;
  contextPrompt: string;
}

export const ARENA_PROBE_WORDS: ProbeTokenRef[] = [
  { id: 'geo', word: 'Hà Nội', contextPrompt: 'Hà Nội' },
  { id: 'code', word: 'Python', contextPrompt: 'Python' },
  { id: 'food', word: 'Phở', contextPrompt: 'Phở' },
  { id: 'math', word: 'Đạo hàm', contextPrompt: 'Đạo hàm' },
];

export const ARENA_GEN_PROMPTS: string[] = [
  'Hà Nội là',
  'Python là',
  'Phở bò là',
  'Tam giác vuông',
];

interface AdamParam {
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
}

interface Sample {
  curId: number;
  contextIds: number[];
  target: number;
  domain: DomainId; // Observer-only tag; NEVER accessed by forward/backward
  tokenText: string;
}

export interface NextPredictionTop3 {
  word: string;
  prob: number;
}

export interface DenseProbeInspection {
  activations: number[]; // 18 normalized neuron activation magnitudes in [0..1]
  top3Next: NextPredictionTop3[];
}

export interface MoeProbeInspection {
  topExperts: [number, number];
  weights: [number, number];
  gateProbs: number[];
  top3Next: NextPredictionTop3[];
}

export interface GenerationSample {
  prompt: string;
  continuation: string;
}

export interface DualEpochSnapshot {
  epoch: number;
  step: number;
  auxLossEnabled: boolean;
  stageLabel: string;
  dense: {
    ceLoss: number;
    totalParams: number;
    ffnParams: number;
    activeFfnNeurons: number;
    totalFfnNeurons: number;
    probeByDomain: Record<DomainId, DenseProbeInspection>;
    generations: GenerationSample[];
  };
  denseBig: {
    ceLoss: number;
    totalParams: number;
    ffnParams: number;
    activeFfnNeurons: number;
    totalFfnNeurons: number;
    generations: GenerationSample[];
  };
  moe: {
    ceLoss: number;
    auxLoss: number;
    rawImbalance: number;
    effectiveExperts: number;
    specializationPct: number;
    totalParams: number;
    ffnParams: number;
    activeFfnNeurons: number;
    totalFfnNeurons: number;
    experts: ExpertEmergenceSnapshot[];
    probeByDomain: Record<DomainId, MoeProbeInspection>;
    generations: GenerationSample[];
  };
}

export interface DualArenaUpdateMessage {
  type: 'DUAL_ARENA_UPDATE';
  running: boolean;
  seed: number;
  auxLossEnabled: boolean;
  latestEpoch: number;
  viewedEpoch: number;
  maxEpochs: number;
  vocabSize: number;
  totalSamples: number;
  current: DualEpochSnapshot;
  history: Array<{
    epoch: number;
    auxLossEnabled: boolean;
    denseCeLoss: number;
    denseBigCeLoss: number;
    moeCeLoss: number;
    moeSpecPct: number;
    moeEffExperts: number;
  }>;
}

export type DualArenaInMessage =
  | { type: 'INIT' }
  | { type: 'START' }
  | { type: 'PAUSE' }
  | { type: 'RUN_ONE_EPOCH' }
  | { type: 'SELECT_EPOCH'; epoch: number }
  | { type: 'RESEED'; seed?: number }
  | { type: 'SET_AUX_LOSS'; enabled: boolean };

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface SharedInitWeights {
  E: Float32Array;
  Wout: Float32Array;
  bout: Float32Array;
}

function createAdamParam(size: number, scale: number, randn: () => number): AdamParam {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = randn() * scale;
  return {
    data,
    grad: new Float32Array(size),
    m: new Float32Array(size),
    v: new Float32Array(size),
  };
}

function cloneAdamParam(src: Float32Array): AdamParam {
  return {
    data: new Float32Array(src),
    grad: new Float32Array(src.length),
    m: new Float32Array(src.length),
    v: new Float32Array(src.length),
  };
}

function computeEffectiveExpertsEntropy(loads: number[]): number {
  let entropy = 0;
  for (const p of loads) {
    if (p > 1e-9) entropy -= p * Math.log(p);
  }
  return Number(Math.exp(entropy).toFixed(2));
}

function adamStep(p: AdamParam, t: number, lr: number): void {
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;
  const bc1 = 1 - Math.pow(b1, t);
  const bc2 = 1 - Math.pow(b2, t);
  for (let i = 0; i < p.data.length; i++) {
    const g = p.grad[i];
    if (g === 0 && p.m[i] === 0) continue;
    p.m[i] = b1 * p.m[i] + (1 - b1) * g;
    p.v[i] = b2 * p.v[i] + (1 - b2) * g * g;
    const mHat = p.m[i] / bc1;
    const vHat = p.v[i] / bc2;
    p.data[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
  }
}

function forwardMlp(
  x: Float32Array,
  inDim: number,
  W1: Float32Array,
  b1: Float32Array,
  W2: Float32Array,
  b2: Float32Array,
  hDim: number
): { hidden: Float32Array; out: Float32Array } {
  const hidden = new Float32Array(hDim);
  for (let i = 0; i < hDim; i++) {
    let sum = b1[i];
    const off = i * inDim;
    for (let j = 0; j < inDim; j++) sum += W1[off + j] * x[j];
    hidden[i] = Math.tanh(sum);
  }
  const out = new Float32Array(inDim);
  for (let i = 0; i < inDim; i++) {
    let sum = b2[i];
    const off = i * hDim;
    for (let j = 0; j < hDim; j++) sum += W2[off + j] * hidden[j];
    out[i] = sum;
  }
  return { hidden, out };
}

function backpropMlp(
  x: Float32Array,
  inDim: number,
  hidden: Float32Array,
  dh: Float32Array,
  scale: number,
  W1: AdamParam,
  b1: AdamParam,
  W2: AdamParam,
  b2: AdamParam,
  hDim: number,
  dxOut: Float32Array
): void {
  const dHidden = new Float32Array(hDim);
  for (let i = 0; i < inDim; i++) {
    const gOut = dh[i] * scale;
    b2.grad[i] += gOut;
    const off = i * hDim;
    for (let j = 0; j < hDim; j++) {
      W2.grad[off + j] += gOut * hidden[j];
      dHidden[j] += gOut * W2.data[off + j];
    }
  }
  for (let j = 0; j < hDim; j++) {
    const dz1 = dHidden[j] * (1 - hidden[j] * hidden[j]);
    b1.grad[j] += dz1;
    const off = j * inDim;
    for (let i = 0; i < inDim; i++) {
      W1.grad[off + i] += dz1 * x[i];
      dxOut[i] += dz1 * W1.data[off + i];
    }
  }
}

function computeSoftmaxProbs(
  h: Float32Array,
  inDim: number,
  V: number,
  Wout: Float32Array,
  bout: Float32Array
): Float32Array {
  const logits = new Float32Array(V);
  let maxLogit = -Infinity;
  for (let v = 0; v < V; v++) {
    let sum = bout[v];
    const off = v * inDim;
    for (let i = 0; i < inDim; i++) sum += Wout[off + i] * h[i];
    logits[v] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }
  const probs = new Float32Array(V);
  let sumExp = 0;
  for (let v = 0; v < V; v++) {
    const ex = Math.exp(logits[v] - maxLogit);
    probs[v] = ex;
    sumExp += ex;
  }
  const invSum = 1 / sumExp;
  for (let v = 0; v < V; v++) probs[v] *= invSum;
  return probs;
}

function buildContextVecFromEmb(
  E: Float32Array,
  D: number,
  curId: number,
  contextIds: number[]
): Float32Array {
  const x = new Float32Array(D);
  const curOff = curId * D;
  if (contextIds.length === 0) {
    for (let d = 0; d < D; d++) x[d] = E[curOff + d];
    return x;
  }
  for (let d = 0; d < D; d++) x[d] = 0.4 * E[curOff + d];
  const ctxScale = 0.6 / contextIds.length;
  for (let c = 0; c < contextIds.length; c++) {
    const off = contextIds[c] * D;
    for (let d = 0; d < D; d++) x[d] += ctxScale * E[off + d];
  }
  return x;
}

function backpropContextVecIntoEmb(
  EGrad: Float32Array,
  D: number,
  curId: number,
  contextIds: number[],
  dx: Float32Array
): void {
  const curOff = curId * D;
  if (contextIds.length === 0) {
    for (let d = 0; d < D; d++) EGrad[curOff + d] += dx[d];
    return;
  }
  for (let d = 0; d < D; d++) EGrad[curOff + d] += 0.4 * dx[d];
  const ctxScale = 0.6 / contextIds.length;
  for (let c = 0; c < contextIds.length; c++) {
    const off = contextIds[c] * D;
    for (let d = 0; d < D; d++) EGrad[off + d] += ctxScale * dx[d];
  }
}

class DenseCore {
  public readonly H: number;
  public readonly ffnParams: number;
  public readonly totalParams: number;

  private E: AdamParam;
  private W1: AdamParam;
  private b1: AdamParam;
  private W2: AdamParam;
  private b2: AdamParam;
  private Wout: AdamParam;
  private bout: AdamParam;
  private step = 0;

  constructor(
    private readonly V: number,
    private readonly D: number,
    H: number,
    shared: SharedInitWeights,
    randn: () => number
  ) {
    this.H = H;
    this.E = cloneAdamParam(shared.E);
    this.W1 = createAdamParam(this.H * D, Math.sqrt(1.2 / D), randn);
    this.b1 = createAdamParam(this.H, 0.02, randn);
    this.W2 = createAdamParam(D * this.H, Math.sqrt(1.2 / this.H), randn);
    this.b2 = createAdamParam(D, 0.01, randn);
    this.Wout = cloneAdamParam(shared.Wout);
    this.bout = cloneAdamParam(shared.bout);

    this.ffnParams =
      this.W1.data.length + this.b1.data.length + this.W2.data.length + this.b2.data.length;
    this.totalParams =
      this.E.data.length + this.ffnParams + this.Wout.data.length + this.bout.data.length;
  }

  private params(): AdamParam[] {
    return [this.E, this.W1, this.b1, this.W2, this.b2, this.Wout, this.bout];
  }

  public trainBatch(batch: Sample[], baseLr: number): void {
    for (const p of this.params()) p.grad.fill(0);
    const invB = 1 / batch.length;
    const dh = new Float32Array(this.D);
    const dx = new Float32Array(this.D);

    for (const s of batch) {
      const x = buildContextVecFromEmb(this.E.data, this.D, s.curId, s.contextIds);
      const { hidden, out } = forwardMlp(
        x,
        this.D,
        this.W1.data,
        this.b1.data,
        this.W2.data,
        this.b2.data,
        this.H
      );
      const probs = computeSoftmaxProbs(out, this.D, this.V, this.Wout.data, this.bout.data);

      dh.fill(0);
      for (let v = 0; v < this.V; v++) {
        const dz = (probs[v] - (v === s.target ? 1 : 0)) * invB;
        this.bout.grad[v] += dz;
        const off = v * this.D;
        for (let i = 0; i < this.D; i++) {
          this.Wout.grad[off + i] += dz * out[i];
          dh[i] += dz * this.Wout.data[off + i];
        }
      }

      dx.fill(0);
      backpropMlp(x, this.D, hidden, dh, 1.0, this.W1, this.b1, this.W2, this.b2, this.H, dx);
      backpropContextVecIntoEmb(this.E.grad, this.D, s.curId, s.contextIds, dx);
    }

    this.step += 1;
    const lr = baseLr * (0.25 + 0.75 * Math.exp(-this.step / 180));
    for (const p of this.params()) adamStep(p, this.step, lr);
  }

  public evalLoss(samples: Sample[]): number {
    let total = 0;
    for (const s of samples) {
      const x = buildContextVecFromEmb(this.E.data, this.D, s.curId, s.contextIds);
      const { out } = forwardMlp(
        x,
        this.D,
        this.W1.data,
        this.b1.data,
        this.W2.data,
        this.b2.data,
        this.H
      );
      const probs = computeSoftmaxProbs(out, this.D, this.V, this.Wout.data, this.bout.data);
      total += -Math.log(Math.max(1e-7, probs[s.target]));
    }
    return Number((total / Math.max(1, samples.length)).toFixed(4));
  }

  public inspect(
    ids: number[],
    idToWord: string[]
  ): DenseProbeInspection {
    const curId = ids[ids.length - 1];
    const contextIds = ids.slice(0, ids.length - 1);
    const x = buildContextVecFromEmb(this.E.data, this.D, curId, contextIds);
    const { hidden, out } = forwardMlp(
      x,
      this.D,
      this.W1.data,
      this.b1.data,
      this.W2.data,
      this.b2.data,
      this.H
    );
    const probs = computeSoftmaxProbs(out, this.D, this.V, this.Wout.data, this.bout.data);

    const activations = Array.from(hidden, (v) => Number(Math.abs(v).toFixed(3)));
    const top3Next = Array.from(probs, (prob, id) => ({
      word: idToWord[id],
      prob: Number(prob.toFixed(3)),
    }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);

    return { activations, top3Next };
  }

  public generate(promptIds: number[], idToWord: string[], maxNew = 6): string {
    const seq = [...promptIds];
    for (let g = 0; g < maxNew; g++) {
      const curId = seq[seq.length - 1];
      const contextIds = seq.slice(0, seq.length - 1);
      const x = buildContextVecFromEmb(this.E.data, this.D, curId, contextIds);
      const { out } = forwardMlp(
        x,
        this.D,
        this.W1.data,
        this.b1.data,
        this.W2.data,
        this.b2.data,
        this.H
      );
      const probs = computeSoftmaxProbs(out, this.D, this.V, this.Wout.data, this.bout.data);
      let bestId = 1;
      let bestP = -1;
      for (let v = 0; v < this.V; v++) {
        if (v === 0 || v === 2) continue;
        if (probs[v] > bestP) {
          bestP = probs[v];
          bestId = v;
        }
      }
      if (bestId === 1) break;
      seq.push(bestId);
    }
    const cont = seq
      .slice(promptIds.length)
      .map((id) => idToWord[id])
      .join(' ');
    return cont || '(kết thúc câu)';
  }
}

class MoECore {
  public readonly N = 4;
  public readonly K = 2;
  public readonly H_EXP = 7;
  public readonly H_SHARED = 4;
  public readonly activeFfnNeurons = 2 * 7 + 4; // 18 active FFN neurons
  public readonly totalFfnNeurons = 4 * 7 + 4; // 32 total FFN neurons
  public readonly ffnParams: number;
  public readonly totalParams: number;

  private readonly SHARED_SCALE = 1.0;
  private readonly ROUTER_LR_MULT = 2.6;
  private readonly ROUTER_TEMP = 0.48;
  private readonly AUX_ALPHA = 0.2;

  private E: AdamParam;
  private Wgate: AdamParam;
  private bgate: AdamParam;
  private W1e: AdamParam[] = [];
  private b1e: AdamParam[] = [];
  private W2e: AdamParam[] = [];
  private b2e: AdamParam[] = [];
  private W1s: AdamParam;
  private b1s: AdamParam;
  private W2s: AdamParam;
  private b2s: AdamParam;
  private Wout: AdamParam;
  private bout: AdamParam;
  private step = 0;

  constructor(
    private readonly V: number,
    private readonly D: number,
    shared: SharedInitWeights,
    randn: () => number
  ) {
    this.E = cloneAdamParam(shared.E);
    this.Wgate = createAdamParam(this.N * D, 0.18, randn);
    this.bgate = createAdamParam(this.N, 0.01, randn);

    for (let e = 0; e < this.N; e++) {
      this.W1e.push(createAdamParam(this.H_EXP * D, Math.sqrt(1.2 / D), randn));
      this.b1e.push(createAdamParam(this.H_EXP, 0.02, randn));
      this.W2e.push(createAdamParam(D * this.H_EXP, Math.sqrt(1.2 / this.H_EXP), randn));
      this.b2e.push(createAdamParam(D, 0.01, randn));
    }

    this.W1s = createAdamParam(this.H_SHARED * D, Math.sqrt(1.2 / D), randn);
    this.b1s = createAdamParam(this.H_SHARED, 0.01, randn);
    this.W2s = createAdamParam(D * this.H_SHARED, Math.sqrt(1.2 / this.H_SHARED), randn);
    this.b2s = createAdamParam(D, 0.01, randn);

    this.Wout = cloneAdamParam(shared.Wout);
    this.bout = cloneAdamParam(shared.bout);

    this.ffnParams =
      this.Wgate.data.length +
      this.bgate.data.length +
      this.W1e.reduce((s, p) => s + p.data.length, 0) +
      this.b1e.reduce((s, p) => s + p.data.length, 0) +
      this.W2e.reduce((s, p) => s + p.data.length, 0) +
      this.b2e.reduce((s, p) => s + p.data.length, 0) +
      this.W1s.data.length +
      this.b1s.data.length +
      this.W2s.data.length +
      this.b2s.data.length;
    this.totalParams =
      this.E.data.length + this.ffnParams + this.Wout.data.length + this.bout.data.length;
  }

  private params(): AdamParam[] {
    return [
      this.E,
      this.Wgate,
      this.bgate,
      ...this.W1e,
      ...this.b1e,
      ...this.W2e,
      ...this.b2e,
      this.W1s,
      this.b1s,
      this.W2s,
      this.b2s,
      this.Wout,
      this.bout,
    ];
  }

  private route(x: Float32Array): {
    gateProbs: number[];
    topExperts: [number, number];
    weights: [number, number];
    topSum: number;
  } {
    const invT = 1 / this.ROUTER_TEMP;
    const logits = [0, 0, 0, 0];
    let maxL = -Infinity;
    for (let e = 0; e < this.N; e++) {
      let s = this.bgate.data[e];
      const off = e * this.D;
      for (let j = 0; j < this.D; j++) s += this.Wgate.data[off + j] * x[j];
      logits[e] = s * invT;
      if (logits[e] > maxL) maxL = logits[e];
    }

    const probs = [0, 0, 0, 0];
    let sumExp = 0;
    for (let e = 0; e < this.N; e++) {
      const ex = Math.exp(logits[e] - maxL);
      probs[e] = ex;
      sumExp += ex;
    }
    for (let e = 0; e < this.N; e++) probs[e] /= sumExp;

    let e0 = 0;
    let e1 = 1;
    if (probs[e1] > probs[e0]) {
      e0 = 1;
      e1 = 0;
    }
    for (let e = 2; e < this.N; e++) {
      if (probs[e] > probs[e0]) {
        e1 = e0;
        e0 = e;
      } else if (probs[e] > probs[e1]) {
        e1 = e;
      }
    }
    const topSum = Math.max(1e-7, probs[e0] + probs[e1]);
    return {
      gateProbs: probs,
      topExperts: [e0, e1],
      weights: [probs[e0] / topSum, probs[e1] / topSum],
      topSum,
    };
  }

  private forwardHidden(x: Float32Array): {
    h: Float32Array;
    gateProbs: number[];
    topExperts: [number, number];
    weights: [number, number];
  } {
    const { gateProbs, topExperts, weights } = this.route(x);
    const [eA, eB] = topExperts;
    const [wA, wB] = weights;
    const expA = forwardMlp(x, this.D, this.W1e[eA].data, this.b1e[eA].data, this.W2e[eA].data, this.b2e[eA].data, this.H_EXP);
    const expB = forwardMlp(x, this.D, this.W1e[eB].data, this.b1e[eB].data, this.W2e[eB].data, this.b2e[eB].data, this.H_EXP);
    const expS = forwardMlp(x, this.D, this.W1s.data, this.b1s.data, this.W2s.data, this.b2s.data, this.H_SHARED);

    const h = new Float32Array(this.D);
    for (let i = 0; i < this.D; i++) {
      h[i] = wA * expA.out[i] + wB * expB.out[i] + this.SHARED_SCALE * expS.out[i];
    }
    return { h, gateProbs, topExperts, weights };
  }

  public trainBatch(batch: Sample[], baseLr: number, auxLossEnabled: boolean): void {
    for (const p of this.params()) p.grad.fill(0);
    const invB = 1 / batch.length;
    const counts = [0, 0, 0, 0];
    const sumProbs = [0, 0, 0, 0];
    const caches: Array<{
      s: Sample;
      x: Float32Array;
      gateProbs: number[];
      dP_ce: [number, number, number, number];
    }> = [];

    const dh = new Float32Array(this.D);
    const dx = new Float32Array(this.D);

    for (const s of batch) {
      const x = buildContextVecFromEmb(this.E.data, this.D, s.curId, s.contextIds);
      const { gateProbs, topExperts, weights, topSum } = this.route(x);
      const [eA, eB] = topExperts;
      const [wA, wB] = weights;

      counts[eA] += 1;
      counts[eB] += 1;
      for (let e = 0; e < this.N; e++) sumProbs[e] += gateProbs[e];

      const expA = forwardMlp(x, this.D, this.W1e[eA].data, this.b1e[eA].data, this.W2e[eA].data, this.b2e[eA].data, this.H_EXP);
      const expB = forwardMlp(x, this.D, this.W1e[eB].data, this.b1e[eB].data, this.W2e[eB].data, this.b2e[eB].data, this.H_EXP);
      const expS = forwardMlp(x, this.D, this.W1s.data, this.b1s.data, this.W2s.data, this.b2s.data, this.H_SHARED);

      const h = new Float32Array(this.D);
      for (let i = 0; i < this.D; i++) {
        h[i] = wA * expA.out[i] + wB * expB.out[i] + this.SHARED_SCALE * expS.out[i];
      }

      const probs = computeSoftmaxProbs(h, this.D, this.V, this.Wout.data, this.bout.data);

      dh.fill(0);
      for (let v = 0; v < this.V; v++) {
        const dz = (probs[v] - (v === s.target ? 1 : 0)) * invB;
        this.bout.grad[v] += dz;
        const off = v * this.D;
        for (let i = 0; i < this.D; i++) {
          this.Wout.grad[off + i] += dz * h[i];
          dh[i] += dz * this.Wout.data[off + i];
        }
      }

      let dotA = 0;
      let dotB = 0;
      for (let i = 0; i < this.D; i++) {
        dotA += dh[i] * expA.out[i];
        dotB += dh[i] * expB.out[i];
      }
      // Quotient rule gradient through wA = pA / (pA + pB), wB = pB / (pA + pB)
      const dP_ce: [number, number, number, number] = [0, 0, 0, 0];
      dP_ce[eA] = (wB * (dotA - dotB)) / topSum;
      dP_ce[eB] = (wA * (dotB - dotA)) / topSum;
      caches.push({ s, x, gateProbs, dP_ce });

      dx.fill(0);
      backpropMlp(x, this.D, expA.hidden, dh, wA, this.W1e[eA], this.b1e[eA], this.W2e[eA], this.b2e[eA], this.H_EXP, dx);
      backpropMlp(x, this.D, expB.hidden, dh, wB, this.W1e[eB], this.b1e[eB], this.W2e[eB], this.b2e[eB], this.H_EXP, dx);
      backpropMlp(x, this.D, expS.hidden, dh, this.SHARED_SCALE, this.W1s, this.b1s, this.W2s, this.b2s, this.H_SHARED, dx);
      backpropContextVecIntoEmb(this.E.grad, this.D, s.curId, s.contextIds, dx);
    }

    const denom = batch.length * (this.K / this.N);
    const ce = counts.map((c) => c / denom);
    const alpha = auxLossEnabled ? this.AUX_ALPHA : 0;

    const invT = 1 / this.ROUTER_TEMP;
    const dxRouter = new Float32Array(this.D);
    for (const item of caches) {
      const dP = [0, 0, 0, 0];
      for (let e = 0; e < this.N; e++) dP[e] = item.dP_ce[e] + alpha * ce[e] * invB;
      let dot = 0;
      for (let e = 0; e < this.N; e++) dot += item.gateProbs[e] * dP[e];
      dxRouter.fill(0);
      for (let e = 0; e < this.N; e++) {
        const ds = item.gateProbs[e] * (dP[e] - dot) * invT;
        this.bgate.grad[e] += ds;
        const off = e * this.D;
        for (let j = 0; j < this.D; j++) {
          this.Wgate.grad[off + j] += ds * item.x[j];
          dxRouter[j] += ds * this.Wgate.data[off + j];
        }
      }
      backpropContextVecIntoEmb(this.E.grad, this.D, item.s.curId, item.s.contextIds, dxRouter);
    }

    this.step += 1;
    const lr = baseLr * (0.25 + 0.75 * Math.exp(-this.step / 180));
    for (const p of this.params()) {
      const paramLr = p === this.Wgate || p === this.bgate ? lr * this.ROUTER_LR_MULT : lr;
      adamStep(p, this.step, paramLr);
    }
  }

  public evaluate(samples: Sample[], auxLossEnabled: boolean): {
    ceLoss: number;
    auxLoss: number;
    rawImbalance: number;
    effectiveExperts: number;
    specializationPct: number;
    experts: ExpertEmergenceSnapshot[];
  } {
    let totalCe = 0;
    const counts = [0, 0, 0, 0];
    const sumProbs = [0, 0, 0, 0];
    const expertSlotTotals = [0, 0, 0, 0];
    const domainCounts: Array<Record<DomainId, number>> = Array.from({ length: this.N }, () => ({
      geo: 0,
      code: 0,
      food: 0,
      math: 0,
    }));
    const tokenStats: Array<
      Map<string, { text: string; domain: DomainId; weightSum: number; count: number }>
    > = Array.from({ length: this.N }, () => new Map());

    for (const s of samples) {
      const x = buildContextVecFromEmb(this.E.data, this.D, s.curId, s.contextIds);
      const { h, gateProbs, topExperts, weights } = this.forwardHidden(x);
      const probs = computeSoftmaxProbs(h, this.D, this.V, this.Wout.data, this.bout.data);
      totalCe += -Math.log(Math.max(1e-7, probs[s.target]));

      counts[topExperts[0]] += 1;
      counts[topExperts[1]] += 1;
      for (let e = 0; e < this.N; e++) sumProbs[e] += gateProbs[e];

      for (let slot = 0; slot < 2; slot++) {
        const e = topExperts[slot];
        const w = weights[slot];
        expertSlotTotals[e] += w;
        domainCounts[e][s.domain] += w;
        const key = `${s.tokenText}|${s.domain}`;
        const prev = tokenStats[e].get(key);
        if (prev) {
          prev.weightSum += w;
          prev.count += 1;
        } else {
          tokenStats[e].set(key, { text: s.tokenText, domain: s.domain, weightSum: w, count: 1 });
        }
      }
    }

    const nSamples = Math.max(1, samples.length);
    const ce = counts.map((c) => c / (nSamples * (this.K / this.N)));
    const meanP = sumProbs.map((s) => s / nSamples);
    const rawImbalance = ce.reduce((acc, c, e) => acc + c * meanP[e], 0);
    const auxLoss = auxLossEnabled ? Number((this.AUX_ALPHA * rawImbalance).toFixed(4)) : 0;

    const grandTotal = expertSlotTotals.reduce((a, b) => a + b, 0) || 1;
    const loads = expertSlotTotals.map((t) => t / grandTotal);
    const effectiveExperts = computeEffectiveExpertsEntropy(loads);

    const experts: ExpertEmergenceSnapshot[] = [];
    let weightedPurity = 0;
    for (let e = 0; e < this.N; e++) {
      const tot = expertSlotTotals[e];
      const shares: Record<DomainId, number> = { geo: 0, code: 0, food: 0, math: 0 };
      let bestDom: DomainId | null = null;
      let maxShare = 0;
      if (tot > 1e-5) {
        for (const d of DOMAIN_ORDER) {
          const sh = domainCounts[e][d] / tot;
          shares[d] = Number(sh.toFixed(3));
          if (sh > maxShare) {
            maxShare = sh;
            bestDom = d;
          }
        }
      }
      weightedPurity += loads[e] * maxShare;
      const topTokens = Array.from(tokenStats[e].values())
        .sort((a, b) => b.weightSum - a.weightSum)
        .slice(0, 4)
        .map((it) => ({
          text: it.text,
          domain: it.domain,
          avgWeight: Number((it.weightSum / Math.max(1, it.count)).toFixed(2)),
          count: it.count,
        }));
      experts.push({
        expertIdx: e,
        loadPct: Number((loads[e] * 100).toFixed(1)),
        domainShares: shares,
        dominantDomain: tot > 1e-5 ? bestDom : null,
        purity: Number(maxShare.toFixed(3)),
        topTokens,
      });
    }

    const specializationPct = Math.round(
      Math.max(0, Math.min(1, (weightedPurity - 0.25) / 0.75)) * 100
    );

    return {
      ceLoss: Number((totalCe / nSamples).toFixed(4)),
      auxLoss,
      rawImbalance: Number(rawImbalance.toFixed(3)),
      effectiveExperts,
      specializationPct,
      experts,
    };
  }

  public inspect(ids: number[], idToWord: string[]): MoeProbeInspection {
    const curId = ids[ids.length - 1];
    const contextIds = ids.slice(0, ids.length - 1);
    const x = buildContextVecFromEmb(this.E.data, this.D, curId, contextIds);
    const { h, gateProbs, topExperts, weights } = this.forwardHidden(x);
    const probs = computeSoftmaxProbs(h, this.D, this.V, this.Wout.data, this.bout.data);

    const top3Next = Array.from(probs, (prob, id) => ({
      word: idToWord[id],
      prob: Number(prob.toFixed(3)),
    }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);

    return {
      topExperts,
      weights: [Number(weights[0].toFixed(2)), Number(weights[1].toFixed(2))],
      gateProbs: gateProbs.map((p) => Number(p.toFixed(2))),
      top3Next,
    };
  }

  public generate(promptIds: number[], idToWord: string[], maxNew = 6): string {
    const seq = [...promptIds];
    for (let g = 0; g < maxNew; g++) {
      const curId = seq[seq.length - 1];
      const contextIds = seq.slice(0, seq.length - 1);
      const x = buildContextVecFromEmb(this.E.data, this.D, curId, contextIds);
      const { h } = this.forwardHidden(x);
      const probs = computeSoftmaxProbs(h, this.D, this.V, this.Wout.data, this.bout.data);
      let bestId = 1;
      let bestP = -1;
      for (let v = 0; v < this.V; v++) {
        if (v === 0 || v === 2) continue;
        if (probs[v] > bestP) {
          bestP = probs[v];
          bestId = v;
        }
      }
      if (bestId === 1) break;
      seq.push(bestId);
    }
    const cont = seq
      .slice(promptIds.length)
      .map((id) => idToWord[id])
      .join(' ');
    return cont || '(kết thúc câu)';
  }
}

export class DualArenaEngine {
  public readonly MAX_EPOCHS = 25;
  public readonly BATCH_SIZE = 64;
  private readonly D = 24;
  private readonly BASE_LR = 0.032;

  private wordToId = new Map<string, number>();
  private idToWord: string[] = [];
  private V = 0;
  private allSamples: Sample[] = [];

  private dense!: DenseCore;
  private denseBig!: DenseCore;
  private moe!: MoECore;

  public seed = 42;
  public auxLossEnabled = true;
  public epoch = 0;
  public step = 0;
  public snapshots: DualEpochSnapshot[] = [];
  public viewedEpoch = 0;

  constructor(seed = 42) {
    this.buildDataset();
    this.reseed(seed);
  }

  private tokenize(text: string): string[] {
    return text
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0);
  }

  private buildDataset(): void {
    const specials = ['<bos>', '<eos>', '<unk>'];
    for (const s of specials) {
      this.wordToId.set(s, this.idToWord.length);
      this.idToWord.push(s);
    }
    for (const item of CORPUS) {
      for (const word of this.tokenize(item.text)) {
        if (!this.wordToId.has(word)) {
          this.wordToId.set(word, this.idToWord.length);
          this.idToWord.push(word);
        }
      }
    }
    this.V = this.idToWord.length;

    const byDomain: Record<DomainId, typeof CORPUS> = { geo: [], code: [], food: [], math: [] };
    for (const s of CORPUS) byDomain[s.domain].push(s);

    const maxLen = Math.max(...DOMAIN_ORDER.map((d) => byDomain[d].length));
    this.allSamples = [];
    for (let row = 0; row < maxLen; row++) {
      for (const dom of DOMAIN_ORDER) {
        const sent = byDomain[dom][row];
        if (!sent) continue;
        const words = this.tokenize(sent.text);
        const ids = words.map((w) => this.wordToId.get(w) ?? 2);
        for (let i = 0; i < ids.length; i++) {
          this.allSamples.push({
            curId: ids[i],
            contextIds: ids.slice(0, i),
            target: i + 1 < ids.length ? ids[i + 1] : 1,
            domain: dom,
            tokenText: words[i],
          });
        }
      }
    }
  }

  private createRandn(seed: number): () => number {
    const rng = mulberry32(seed);
    return () => {
      const u1 = Math.max(1e-7, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
  }

  public reseed(newSeed: number): void {
    this.seed = newSeed;
    this.epoch = 0;
    this.step = 0;
    this.snapshots = [];
    this.viewedEpoch = 0;

    // Generate shared initial E & LM Head (Wout, bout) once so all 3 models start from the same point.
    const sharedRandn = this.createRandn(newSeed);
    const sharedInit: SharedInitWeights = {
      E: createAdamParam(this.V * this.D, 0.24, sharedRandn).data,
      Wout: createAdamParam(this.V * this.D, Math.sqrt(1.0 / this.D), sharedRandn).data,
      bout: createAdamParam(this.V, 0.01, sharedRandn).data,
    };

    this.dense = new DenseCore(this.V, this.D, 18, sharedInit, this.createRandn(newSeed + 101));
    this.denseBig = new DenseCore(this.V, this.D, 32, sharedInit, this.createRandn(newSeed + 202));
    this.moe = new MoECore(this.V, this.D, sharedInit, this.createRandn(newSeed + 303));

    this.captureSnapshot();
  }

  public trainOneEpoch(): void {
    if (this.epoch >= this.MAX_EPOCHS) return;

    // 1 Epoch = deterministic Fisher-Yates shuffle + complete pass over 100% of samples
    const rng = mulberry32(this.seed * 997 + (this.epoch + 1) * 131);
    const indices = this.allSamples.map((_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }

    for (let off = 0; off < indices.length; off += this.BATCH_SIZE) {
      const batch = indices.slice(off, off + this.BATCH_SIZE).map((idx) => this.allSamples[idx]);
      // All 3 models receive the EXACT SAME shuffled mini-batch in parallel!
      this.dense.trainBatch(batch, this.BASE_LR);
      this.denseBig.trainBatch(batch, this.BASE_LR);
      this.moe.trainBatch(batch, this.BASE_LR, this.auxLossEnabled);
      this.step += 1;
    }

    this.epoch += 1;
    this.captureSnapshot();
    this.viewedEpoch = this.epoch;
  }

  private stageLabelForEpoch(ep: number): string {
    if (ep === 0) return 'Epoch 0: Khởi tạo chung (E & LM Head giống hệt nhau)';
    if (ep <= 3) return `Epoch ${ep}: Học cấu trúc câu & từ nối`;
    if (ep <= 10) return `Epoch ${ep}: Giai đoạn phân bổ tải & hội tụ dần`;
    return `Epoch ${ep}: So sánh 3 mốc — Dense nhỏ, MoE, Dense to`;
  }

  private toIds(text: string): number[] {
    return this.tokenize(text).map((w) => this.wordToId.get(w) ?? 2);
  }

  private captureSnapshot(): void {
    const denseCeLoss = this.dense.evalLoss(this.allSamples);
    const denseBigCeLoss = this.denseBig.evalLoss(this.allSamples);
    const moeEval = this.moe.evaluate(this.allSamples, this.auxLossEnabled);

    const denseProbe = {} as Record<DomainId, DenseProbeInspection>;
    const moeProbe = {} as Record<DomainId, MoeProbeInspection>;
    for (const p of ARENA_PROBE_WORDS) {
      const ids = this.toIds(p.contextPrompt);
      denseProbe[p.id] = this.dense.inspect(ids, this.idToWord);
      moeProbe[p.id] = this.moe.inspect(ids, this.idToWord);
    }

    const denseGens: GenerationSample[] = ARENA_GEN_PROMPTS.map((prompt) => ({
      prompt,
      continuation: this.dense.generate(this.toIds(prompt), this.idToWord, 6),
    }));
    const denseBigGens: GenerationSample[] = ARENA_GEN_PROMPTS.map((prompt) => ({
      prompt,
      continuation: this.denseBig.generate(this.toIds(prompt), this.idToWord, 6),
    }));
    const moeGens: GenerationSample[] = ARENA_GEN_PROMPTS.map((prompt) => ({
      prompt,
      continuation: this.moe.generate(this.toIds(prompt), this.idToWord, 6),
    }));

    this.snapshots.push({
      epoch: this.epoch,
      step: this.step,
      auxLossEnabled: this.auxLossEnabled,
      stageLabel: this.stageLabelForEpoch(this.epoch),
      dense: {
        ceLoss: denseCeLoss,
        totalParams: this.dense.totalParams,
        ffnParams: this.dense.ffnParams,
        activeFfnNeurons: this.dense.H,
        totalFfnNeurons: this.dense.H,
        probeByDomain: denseProbe,
        generations: denseGens,
      },
      denseBig: {
        ceLoss: denseBigCeLoss,
        totalParams: this.denseBig.totalParams,
        ffnParams: this.denseBig.ffnParams,
        activeFfnNeurons: this.denseBig.H,
        totalFfnNeurons: this.denseBig.H,
        generations: denseBigGens,
      },
      moe: {
        ceLoss: moeEval.ceLoss,
        auxLoss: moeEval.auxLoss,
        rawImbalance: moeEval.rawImbalance,
        effectiveExperts: moeEval.effectiveExperts,
        specializationPct: moeEval.specializationPct,
        totalParams: this.moe.totalParams,
        ffnParams: this.moe.ffnParams,
        activeFfnNeurons: this.moe.activeFfnNeurons,
        totalFfnNeurons: this.moe.totalFfnNeurons,
        experts: moeEval.experts,
        probeByDomain: moeProbe,
        generations: moeGens,
      },
    });
  }

  public getUpdateMessage(running: boolean, requestedEpoch?: number): DualArenaUpdateMessage {
    if (typeof requestedEpoch === 'number') {
      this.viewedEpoch = Math.max(0, Math.min(this.snapshots.length - 1, requestedEpoch));
    }
    const idx = Math.max(0, Math.min(this.snapshots.length - 1, this.viewedEpoch));
    return {
      type: 'DUAL_ARENA_UPDATE',
      running,
      seed: this.seed,
      auxLossEnabled: this.auxLossEnabled,
      latestEpoch: this.epoch,
      viewedEpoch: idx,
      maxEpochs: this.MAX_EPOCHS,
      vocabSize: this.V,
      totalSamples: this.allSamples.length,
      current: this.snapshots[idx],
      history: this.snapshots.map((s) => ({
        epoch: s.epoch,
        auxLossEnabled: s.auxLossEnabled,
        denseCeLoss: s.dense.ceLoss,
        denseBigCeLoss: s.denseBig.ceLoss,
        moeCeLoss: s.moe.ceLoss,
        moeSpecPct: s.moe.specializationPct,
        moeEffExperts: s.moe.effectiveExperts,
      })),
    };
  }
}
