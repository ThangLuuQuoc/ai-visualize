/**
 * Real Neural Language Model running inside a browser Web Worker.
 * Architecture: Context-3 Word Embedding (d=24) -> Hidden Tanh (H=48) -> LM Head (V) -> Softmax
 * Training: Real analytical Backpropagation + Adam optimizer on Vietnamese sentences.
 */

const VIETNAMESE_CORPUS: string[] = [
  'Hà Nội là thủ đô của Việt Nam',
  'Hà Nội là thủ đô nghìn năm văn hiến của Việt Nam',
  'thủ đô của Việt Nam là Hà Nội',
  'Việt Nam là quốc gia ở khu vực Đông Nam Á',
  'Việt Nam có thủ đô là Hà Nội và nhiều danh lam thắng cảnh',
  'TP.HCM là trung tâm kinh tế lớn nhất của Việt Nam',
  'TP.HCM là thành phố năng động và hiện đại',
  'Đà Nẵng nổi tiếng với cầu Rồng và bãi biển đẹp',
  'Đà Nẵng là thành phố đáng sống ở miền Trung',
  'Huế là cố đô lịch sử của Việt Nam',
  'Python là ngôn ngữ lập trình phổ biến cho AI',
  'Python là ngôn ngữ dễ đọc và dễ học cho người mới',
  'TypeScript giúp viết code web an toàn và rõ ràng hơn',
  'Three.js dùng để dựng đồ họa 3D ngay trên trình duyệt',
  'Node.js chạy JavaScript phía máy chủ rất nhanh',
  'MiniMind giúp tối ưu tốc độ huấn luyện mạng nhỏ',
  'MiniMind thu gọn từ điển xuống 6400 từ để tiết kiệm bộ nhớ',
  'MiniMind huấn luyện mô hình ngôn ngữ nhỏ trên một GPU phổ thông',
  'Pretrain giúp mô hình đọc sách để học ngữ pháp và kiến thức nền',
  'SFT dạy mô hình cách trả lời câu hỏi theo mẫu hỏi đáp',
  'LoRA đóng băng trọng số gốc để tiết kiệm bộ nhớ VRAM',
  'LoRA chỉ cập nhật hai ma trận nhỏ thay vì toàn bộ mạng',
  'MoE chỉ kích hoạt hai chuyên gia cho mỗi từ đi vào',
  'MoE giúp tăng dung lượng tri thức mà vẫn giữ tốc độ nhanh',
  'GQA cho phép tám đầu truy vấn dùng chung hai tủ hồ sơ',
  'GQA giúp tiết kiệm bộ nhớ khi mô hình sinh câu trả lời',
  'RoPE xoay góc vector theo vị trí của từ trong câu',
  'DPO so sánh trực tiếp câu trả lời hay và câu trả lời dở',
  'GRPO lấy mẫu bốn câu trả lời rồi so điểm với trung bình nhóm',
  'GRPO loại bỏ mô hình giám khảo để giảm một nửa bộ nhớ',
  'Loss đo độ bất ngờ của mô hình khi thấy đáp án đúng',
  'Loss càng thấp nghĩa là mô hình đoán chữ càng chính xác',
  'học lập trình cần thực hành viết code mỗi ngày',
  'người mới nên bắt đầu học lập trình từ ngôn ngữ Python',
  'làm dự án nhỏ giúp hiểu sâu bản chất thuật toán',
  'mạng nơ ron học bằng cách truyền ngược đạo hàm để chỉnh trọng số',
  'trí tuệ nhân tạo đang hỗ trợ lập trình viên làm việc hiệu quả hơn',
  'mô hình ngôn ngữ lớn dự đoán từ tiếp theo dựa trên ngữ cảnh',
  'dữ liệu sạch giúp mô hình nhỏ hội tụ nhanh và chính xác hơn',
  'chưng cất tri thức giúp mô hình nhỏ học kinh nghiệm từ mô hình lớn',
];

// Repeat key knowledge patterns slightly so the tiny corpus has ~160 rich training sentences
const FULL_CORPUS: string[] = [
  ...VIETNAMESE_CORPUS,
  ...VIETNAMESE_CORPUS.slice(0, 20),
  ...VIETNAMESE_CORPUS.slice(10, 35),
  ...VIETNAMESE_CORPUS,
];

interface AdamParam {
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
}

export interface MicroLmUpdateMessage {
  type: 'MICRO_LM_UPDATE';
  step: number;
  epoch: number;
  vocabSize: number;
  totalParams: number;
  batchLoss: number;
  promptText: string;
  targetToken: string;
  targetProb: number;
  tokenSurpriseLoss: number;
  candidates: Array<{ token: string; prob: number; isTarget: boolean }>;
  generatedText: string;
  gradNorms: number[]; // [embedNorm, hiddenNorm, lmHeadNorm]
  stepsPerSec: number;
}

type WorkerInMessage =
  | { type: 'START' }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'RESET' }
  | { type: 'STEP_ONCE' }
  | { type: 'SET_PROMPT'; prompt: string }
  | { type: 'SET_SPEED'; speed: number };

class RealMicroLanguageModel {
  private readonly C = 3; // Context window of 3 preceding words
  private readonly D = 24; // Embedding dimension
  private readonly H = 48; // Hidden dimension
  private readonly LR = 0.018;

  private wordToId = new Map<string, number>();
  private idToWord: string[] = [];
  private V = 0;

  private samples: Array<{ ctx: [number, number, number]; target: number }> = [];
  private E!: AdamParam;
  private W1!: AdamParam;
  private b1!: AdamParam;
  private W2!: AdamParam;
  private b2!: AdamParam;

  public step = 0;
  private cursor = 0;
  public totalParams = 0;
  public promptText = 'Hà Nội là thủ đô của';
  public expectedTarget = 'Việt';

  constructor() {
    this.buildVocabAndSamples();
    this.initWeights();
  }

  private tokenize(text: string): string[] {
    return text
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0);
  }

  private buildVocabAndSamples(): void {
    const specials = ['<bos>', '<eos>', '<unk>'];
    for (const s of specials) {
      this.wordToId.set(s, this.idToWord.length);
      this.idToWord.push(s);
    }

    for (const sentence of FULL_CORPUS) {
      for (const word of this.tokenize(sentence)) {
        if (!this.wordToId.has(word)) {
          this.wordToId.set(word, this.idToWord.length);
          this.idToWord.push(word);
        }
      }
    }

    this.V = this.idToWord.length;
    const bos = 0;
    const eos = 1;

    this.samples = [];
    for (const sentence of FULL_CORPUS) {
      const ids = this.tokenize(sentence).map((w) => this.wordToId.get(w) ?? 2);
      const padded = [bos, bos, bos, ...ids, eos];
      for (let i = this.C; i < padded.length; i++) {
        this.samples.push({
          ctx: [padded[i - 3], padded[i - 2], padded[i - 1]],
          target: padded[i],
        });
      }
    }
  }

  private createParam(size: number, scale: number): AdamParam {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller normal-ish initialization
      const u1 = Math.max(1e-6, Math.random());
      const u2 = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
    }
    return {
      data,
      grad: new Float32Array(size),
      m: new Float32Array(size),
      v: new Float32Array(size),
    };
  }

  public initWeights(): void {
    const inDim = this.C * this.D;
    this.E = this.createParam(this.V * this.D, 0.18);
    this.W1 = this.createParam(this.H * inDim, Math.sqrt(1 / inDim));
    this.b1 = this.createParam(this.H, 0.01);
    this.W2 = this.createParam(this.V * this.H, Math.sqrt(1 / this.H));
    this.b2 = this.createParam(this.V, 0.01);
    this.totalParams =
      this.E.data.length +
      this.W1.data.length +
      this.b1.data.length +
      this.W2.data.length +
      this.b2.data.length;
    this.step = 0;
    this.cursor = 0;
  }

  private forwardSingle(ctx: [number, number, number]): {
    x: Float32Array;
    h: Float32Array;
    probs: Float32Array;
  } {
    const inDim = this.C * this.D;
    const x = new Float32Array(inDim);
    for (let c = 0; c < this.C; c++) {
      const embOffset = ctx[c] * this.D;
      const xOffset = c * this.D;
      for (let d = 0; d < this.D; d++) {
        x[xOffset + d] = this.E.data[embOffset + d];
      }
    }

    const h = new Float32Array(this.H);
    for (let i = 0; i < this.H; i++) {
      let sum = this.b1.data[i];
      const rowOffset = i * inDim;
      for (let j = 0; j < inDim; j++) {
        sum += this.W1.data[rowOffset + j] * x[j];
      }
      h[i] = Math.tanh(sum);
    }

    const logits = new Float32Array(this.V);
    let maxLogit = -Infinity;
    for (let v = 0; v < this.V; v++) {
      let sum = this.b2.data[v];
      const rowOffset = v * this.H;
      for (let i = 0; i < this.H; i++) {
        sum += this.W2.data[rowOffset + i] * h[i];
      }
      logits[v] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    const probs = new Float32Array(this.V);
    let sumExp = 0;
    for (let v = 0; v < this.V; v++) {
      const e = Math.exp(logits[v] - maxLogit);
      probs[v] = e;
      sumExp += e;
    }
    const invSum = 1 / sumExp;
    for (let v = 0; v < this.V; v++) {
      probs[v] *= invSum;
    }

    return { x, h, probs };
  }

  private zeroGrad(p: AdamParam): void {
    p.grad.fill(0);
  }

  private adamStep(p: AdamParam, t: number, lr: number): number {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);
    let sqNorm = 0;

    for (let i = 0; i < p.data.length; i++) {
      const g = p.grad[i];
      sqNorm += g * g;
      p.m[i] = beta1 * p.m[i] + (1 - beta1) * g;
      p.v[i] = beta2 * p.v[i] + (1 - beta2) * g * g;
      const mHat = p.m[i] / bc1;
      const vHat = p.v[i] / bc2;
      p.data[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
    return Math.sqrt(sqNorm);
  }

  public trainBatch(batchSize = 24): { batchLoss: number; gradNorms: number[] } {
    this.zeroGrad(this.E);
    this.zeroGrad(this.W1);
    this.zeroGrad(this.b1);
    this.zeroGrad(this.W2);
    this.zeroGrad(this.b2);

    const inDim = this.C * this.D;
    const invBatch = 1 / batchSize;
    let totalLoss = 0;

    const dh = new Float32Array(this.H);
    const dz1 = new Float32Array(this.H);

    for (let b = 0; b < batchSize; b++) {
      const sample = this.samples[this.cursor % this.samples.length];
      this.cursor = (this.cursor + 1) % this.samples.length;

      const { x, h, probs } = this.forwardSingle(sample.ctx);
      totalLoss += -Math.log(Math.max(1e-7, probs[sample.target]));

      // Analytical Backprop: dL/dz2 = (probs - one_hot(target)) / batchSize
      dh.fill(0);
      for (let v = 0; v < this.V; v++) {
        const dz2 = (probs[v] - (v === sample.target ? 1 : 0)) * invBatch;
        this.b2.grad[v] += dz2;
        const rowOffset = v * this.H;
        for (let i = 0; i < this.H; i++) {
          this.W2.grad[rowOffset + i] += dz2 * h[i];
          dh[i] += dz2 * this.W2.data[rowOffset + i];
        }
      }

      // Backprop through tanh: dz1 = dh * (1 - h^2)
      for (let i = 0; i < this.H; i++) {
        dz1[i] = dh[i] * (1 - h[i] * h[i]);
        this.b1.grad[i] += dz1[i];
      }

      // Backprop into W1 and Embeddings E
      for (let i = 0; i < this.H; i++) {
        const g1 = dz1[i];
        const rowOffset = i * inDim;
        for (let j = 0; j < inDim; j++) {
          this.W1.grad[rowOffset + j] += g1 * x[j];
        }
      }

      for (let c = 0; c < this.C; c++) {
        const embOffset = sample.ctx[c] * this.D;
        const xOffset = c * this.D;
        for (let d = 0; d < this.D; d++) {
          let gx = 0;
          const col = xOffset + d;
          for (let i = 0; i < this.H; i++) {
            gx += dz1[i] * this.W1.data[i * inDim + col];
          }
          this.E.grad[embOffset + d] += gx;
        }
      }
    }

    this.step += 1;
    const lr = this.LR * (0.25 + 0.75 * Math.exp(-this.step / 140));
    const nE = this.adamStep(this.E, this.step, lr);
    const nW1 = this.adamStep(this.W1, this.step, lr);
    this.adamStep(this.b1, this.step, lr);
    const nW2 = this.adamStep(this.W2, this.step, lr);
    this.adamStep(this.b2, this.step, lr);

    return {
      batchLoss: Number((totalLoss * invBatch).toFixed(4)),
      gradNorms: [Number(nE.toFixed(4)), Number(nW1.toFixed(4)), Number(nW2.toFixed(4))],
    };
  }

  public setPrompt(prompt: string): void {
    this.promptText = prompt.trim() || 'Hà Nội là thủ đô của';
    // Infer the most likely ground-truth next word from the corpus if present
    const words = this.tokenize(this.promptText);
    const last2 = words.slice(-2).join(' ').toLowerCase();
    let found = 'Việt';
    for (const s of VIETNAMESE_CORPUS) {
      const sw = this.tokenize(s);
      for (let i = 0; i < sw.length - 1; i++) {
        const pair = sw.slice(Math.max(0, i - 1), i + 1).join(' ').toLowerCase();
        if (pair === last2) {
          found = sw[i + 1];
          break;
        }
      }
    }
    this.expectedTarget = found;
  }

  public inspectPrompt(batchLoss: number, gradNorms: number[], stepsPerSec: number): MicroLmUpdateMessage {
    const words = this.tokenize(this.promptText);
    const ids = words.map((w) => this.wordToId.get(w) ?? 2);
    while (ids.length < 3) ids.unshift(0);
    const ctx: [number, number, number] = [
      ids[ids.length - 3],
      ids[ids.length - 2],
      ids[ids.length - 1],
    ];

    const { probs } = this.forwardSingle(ctx);

    // Top-5 candidates from real softmax distribution
    const indexed = Array.from(probs, (prob, id) => ({
      token: this.idToWord[id],
      prob: Number(prob.toFixed(4)),
      isTarget: this.idToWord[id] === this.expectedTarget,
    }));
    indexed.sort((a, b) => b.prob - a.prob);
    const top5 = indexed.slice(0, 5);

    const targetId = this.wordToId.get(this.expectedTarget);
    const targetProb = targetId !== undefined ? Number(probs[targetId].toFixed(4)) : top5[0].prob;
    const tokenSurpriseLoss = Number((-Math.log(Math.max(1e-4, targetProb))).toFixed(3));

    // Greedy autoregressive generation (up to 7 words) from live trained weights
    const genWords = [...words];
    const genIds = [...ids];
    for (let g = 0; g < 7; g++) {
      const gCtx: [number, number, number] = [
        genIds[genIds.length - 3],
        genIds[genIds.length - 2],
        genIds[genIds.length - 1],
      ];
      const { probs: gProbs } = this.forwardSingle(gCtx);
      let bestId = 1;
      let bestP = -1;
      for (let v = 0; v < this.V; v++) {
        if (v === 0 || v === 2) continue; // skip <bos>/<unk>
        if (gProbs[v] > bestP) {
          bestP = gProbs[v];
          bestId = v;
        }
      }
      if (bestId === 1) break; // <eos>
      genIds.push(bestId);
      genWords.push(this.idToWord[bestId]);
    }

    return {
      type: 'MICRO_LM_UPDATE',
      step: this.step,
      epoch: Math.floor((this.step * 24) / Math.max(1, this.samples.length)) + 1,
      vocabSize: this.V,
      totalParams: this.totalParams,
      batchLoss,
      promptText: this.promptText,
      targetToken: this.expectedTarget,
      targetProb,
      tokenSurpriseLoss,
      candidates: top5,
      generatedText: genWords.join(' '),
      gradNorms,
      stepsPerSec,
    };
  }
}

const model = new RealMicroLanguageModel();
let running = false;
let speed = 1.0;
let timerId: ReturnType<typeof setInterval> | null = null;

function runOneUpdate(): void {
  const t0 = performance.now();
  const { batchLoss, gradNorms } = model.trainBatch(24);
  const dtMs = Math.max(1, performance.now() - t0);
  const msg = model.inspectPrompt(batchLoss, gradNorms, Math.round(1000 / dtMs));
  self.postMessage(msg);
}

function restartTimer(): void {
  if (timerId !== null) clearInterval(timerId);
  const intervalMs = Math.max(80, Math.round(420 / speed));
  timerId = setInterval(() => {
    if (running) {
      runOneUpdate();
    }
  }, intervalMs);
}

self.onmessage = (ev: MessageEvent<WorkerInMessage>) => {
  const cmd = ev.data;
  if (cmd.type === 'START' || cmd.type === 'RESUME') {
    running = true;
    runOneUpdate();
    restartTimer();
  } else if (cmd.type === 'PAUSE') {
    running = false;
  } else if (cmd.type === 'RESET') {
    model.initWeights();
    runOneUpdate();
  } else if (cmd.type === 'STEP_ONCE') {
    running = false;
    runOneUpdate();
  } else if (cmd.type === 'SET_PROMPT') {
    model.setPrompt(cmd.prompt);
    const { batchLoss, gradNorms } = model.trainBatch(1);
    self.postMessage(model.inspectPrompt(batchLoss, gradNorms, 0));
  } else if (cmd.type === 'SET_SPEED') {
    speed = Math.max(0.25, Math.min(3.0, cmd.speed));
    restartTimer();
  }
};
