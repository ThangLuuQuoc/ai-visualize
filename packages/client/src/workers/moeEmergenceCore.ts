/**
 * Real 4-Expert Top-2 + 1 Shared Expert MoE Language Model (Pure TypeScript Core).
 *
 * Scientific guarantee:
 * - The model ONLY sees raw word IDs (causal context pooling -> next-word target).
 * - Domain tags (`geo`, `code`, `food`, `math`) are strictly Observer-only and are NEVER
 *   read by `routeInput()`, `trainOneStep()`, or any gradient computation.
 * - Specialization emerges purely from symmetry breaking and co-adaptation via analytical
 *   Backpropagation through Cross-Entropy Loss + MiniMind2 seq_aux Load-Balancing Loss.
 */

export type DomainId = 'geo' | 'code' | 'food' | 'math';

export const DOMAIN_ORDER: DomainId[] = ['geo', 'code', 'food', 'math'];

export const DOMAIN_META: Record<DomainId, { label: string; short: string; color: string }> = {
  geo: { label: 'Địa lý', short: '🗺️ Địa lý', color: '#38bdf8' },
  code: { label: 'Lập trình / AI', short: '💻 Lập trình', color: '#c084fc' },
  food: { label: 'Ẩm thực', short: '🍜 Ẩm thực', color: '#fbbf24' },
  math: { label: 'Toán / Khoa học', short: '🔢 Toán học', color: '#34d399' },
};

export interface RawSentence {
  domain: DomainId;
  text: string;
}

// Distributional Vietnamese corpus: words within each domain share natural bridge terms
// (miền/biển/đảo/sông/núi; code/hàm/biến/dữ_liệu/server/lỗi; món/ăn/nước/thơm/giòn/chấm;
// tính/công_thức/bằng/góc/độ/số/lực), while connectors (là, có, và, của, với, trong, rất)
// appear across all four domains.
export const CORPUS: RawSentence[] = [
  // 🗺️ Địa lý (16 sentences)
  { domain: 'geo', text: 'Hà Nội là thủ đô ở miền Bắc và sông Hồng chảy qua' },
  { domain: 'geo', text: 'Đà Nẵng là thành phố biển miền Trung có cầu Rồng và bãi biển đẹp' },
  { domain: 'geo', text: 'Huế nằm bên dòng sông Hương ở miền Trung có phố cổ đẹp' },
  { domain: 'geo', text: 'Sài Gòn là thành phố lớn ở miền Nam với dòng sông rộng' },
  { domain: 'geo', text: 'Vịnh Hạ Long có hàng nghìn hòn đảo đá giữa biển xanh miền Bắc' },
  { domain: 'geo', text: 'Sapa là vùng núi cao miền Bắc có đỉnh núi Fansipan và rừng thông' },
  { domain: 'geo', text: 'Cần Thơ ở miền Tây có chợ nổi trên sông và nhiều kênh rạch' },
  { domain: 'geo', text: 'Phú Quốc là đảo ngọc giữa biển xanh ở miền Nam có bãi biển đẹp' },
  { domain: 'geo', text: 'Nha Trang nổi tiếng với đảo xanh và bãi biển đẹp ở miền Trung' },
  { domain: 'geo', text: 'Đà Lạt là thành phố núi ở miền Trung có rừng thông khí hậu mát' },
  { domain: 'geo', text: 'Hội An có phố cổ ven sông ở miền Trung và đèn lồng rực rỡ' },
  { domain: 'geo', text: 'Sông Hồng chảy qua đồng bằng miền Bắc ra biển Đông rộng lớn' },
  { domain: 'geo', text: 'Trường Sơn là dãy núi dài chạy qua miền Trung với nhiều rừng thông' },
  { domain: 'geo', text: 'Côn Đảo là hòn đảo xa giữa biển Đông ở miền Nam rất đẹp' },
  { domain: 'geo', text: 'Quảng Ninh ở miền Bắc nổi tiếng với đảo đá Vịnh Hạ Long' },
  { domain: 'geo', text: 'Đồng bằng sông Cửu Long ở miền Nam có dòng sông dài và kênh rạch' },

  // 💻 Lập trình / AI (16 sentences)
  { domain: 'code', text: 'Python là ngôn ngữ chạy code với hàm def và biến dữ liệu' },
  { domain: 'code', text: 'TypeScript kiểm tra kiểu dữ liệu của biến trước khi chạy code' },
  { domain: 'code', text: 'GPU chạy tính ma trận dữ liệu và cập nhật trọng số rất nhanh' },
  { domain: 'code', text: 'LoRA đóng băng trọng số gốc trong mạng và chỉ sửa hàm nhỏ' },
  { domain: 'code', text: 'Git lưu lại lịch sử sửa lỗi code của ứng dụng trên server' },
  { domain: 'code', text: 'Docker đóng gói ứng dụng và server thành container để chạy ổn định' },
  { domain: 'code', text: 'API trên server trả dữ liệu JSON về cho ứng dụng web xử lý' },
  { domain: 'code', text: 'WebSocket giúp server và ứng dụng truyền dữ liệu hai chiều rất nhanh' },
  { domain: 'code', text: 'Node.js chạy code JavaScript phía server để xử lý dữ liệu và API' },
  { domain: 'code', text: 'SQL dùng lệnh truy vấn bảng dữ liệu trên server và lọc biến' },
  { domain: 'code', text: 'Backprop tính lỗi trên dữ liệu để cập nhật trọng số trong mạng' },
  { domain: 'code', text: 'Linux chạy lệnh shell trên server để kiểm tra lỗi ứng dụng' },
  { domain: 'code', text: 'React dựng giao diện ứng dụng web từ các biến trạng thái và hàm' },
  { domain: 'code', text: 'Three.js chạy code dựng khối ba chiều trong ứng dụng web rất mượt' },
  { domain: 'code', text: 'Adam cập nhật trọng số dựa trên lỗi dữ liệu của từng biến mạng' },
  { domain: 'code', text: 'Tokenizer tách câu thành mã số trước khi mạng xử lý dữ liệu' },

  // 🍜 Ẩm thực (16 sentences)
  { domain: 'food', text: 'Phở bò là món ăn nóng có nước dùng thơm ngon và rau mùi' },
  { domain: 'food', text: 'Bánh mì là món ăn giòn ngon kẹp pate thơm và rau mùi' },
  { domain: 'food', text: 'Bún chả là món ăn thơm ngon nướng ăn kèm nước chấm mắm' },
  { domain: 'food', text: 'Cà phê sữa đá là món uống thơm ngọt pha từ nước sôi' },
  { domain: 'food', text: 'Gỏi cuốn là món ăn ngon có tôm thịt chấm cùng nước tương' },
  { domain: 'food', text: 'Bún bò Huế là món nước nóng thơm sả ớt ăn rất ngon' },
  { domain: 'food', text: 'Cơm tấm là món ăn ngon có sườn nướng thơm ăn kèm nước mắm' },
  { domain: 'food', text: 'Chả giò là món ăn chiên giòn ngon chấm cùng nước mắm chua ngọt' },
  { domain: 'food', text: 'Bánh xèo là món ăn vàng giòn có nhân tôm ăn kèm rau thơm' },
  { domain: 'food', text: 'Mì Quảng là món ăn ngon ở miền Trung có nước dùng thơm lạc rang' },
  { domain: 'food', text: 'Chè đậu xanh là món ăn ngọt mát nấu với nước dừa rất thơm' },
  { domain: 'food', text: 'Bánh cuốn là món ăn nóng mềm rắc hành thơm chấm nước mắm' },
  { domain: 'food', text: 'Hủ tiếu là món nước ngọt thanh ăn nóng với thịt và hành thơm' },
  { domain: 'food', text: 'Bò kho là món ăn nóng hầm mềm thơm quế chấm cùng bánh mì giòn' },
  { domain: 'food', text: 'Trà sen là món uống thơm ngọt pha bằng nước nóng rất dễ chịu' },
  { domain: 'food', text: 'Nem chua rán là món ăn giòn nóng chấm cùng tương ớt rất ngon' },

  // 🔢 Toán / Khoa học (16 sentences)
  { domain: 'math', text: 'Tam giác vuông có góc bằng chín mươi độ theo công thức hình học' },
  { domain: 'math', text: 'Đạo hàm dùng công thức tính tốc độ biến thiên của một hàm số' },
  { domain: 'math', text: 'Số nguyên tố là số tự nhiên chỉ chia hết cho một và chính nó' },
  { domain: 'math', text: 'Trái Đất quay quanh Mặt Trời bằng lực hấp dẫn theo công thức quỹ đạo' },
  { domain: 'math', text: 'Ánh sáng truyền thẳng với tốc độ bằng hằng số trong chân không' },
  { domain: 'math', text: 'Hình tròn có công thức tính chu vi bằng hai pi nhân bán kính' },
  { domain: 'math', text: 'Tích phân dùng công thức tính diện tích dưới đường cong hàm số' },
  { domain: 'math', text: 'Phương trình bậc hai tính nghiệm thực bằng công thức biệt thức delta' },
  { domain: 'math', text: 'Lực hấp dẫn tính bằng công thức khối lượng nhân gia tốc rơi tự do' },
  { domain: 'math', text: 'Nguyên tử có hạt nhân mang điện tích dương và số proton bằng electron' },
  { domain: 'math', text: 'Xác suất của một biến cố luôn có số trị từ không tới một' },
  { domain: 'math', text: 'Vectơ là đại lượng có độ lớn bằng số thực và hướng góc xác định' },
  { domain: 'math', text: 'Nhiệt độ sôi của nước bằng một trăm độ theo công thức vật lý' },
  { domain: 'math', text: 'Ma trận đơn vị là bảng số có đường chéo chính bằng số một' },
  { domain: 'math', text: 'Sóng âm truyền nhanh trong chất rắn với tốc độ đo bằng công thức' },
  { domain: 'math', text: 'Định lý Pytago tính độ dài cạnh tam giác vuông có góc chín mươi độ' },
];

interface AdamParam {
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
}

interface Sample {
  curId: number;
  contextIds: number[]; // Preceding causal token IDs in the sentence
  target: number;
  domain: DomainId; // Observer-only tag; NEVER accessed by forward/backward
  tokenText: string;
}

export interface ExpertEmergenceSnapshot {
  expertIdx: number;
  loadPct: number;
  domainShares: Record<DomainId, number>;
  dominantDomain: DomainId | null;
  purity: number;
  topTokens: Array<{ text: string; domain: DomainId; avgWeight: number; count: number }>;
}

export interface EmergenceHistoryPoint {
  step: number;
  specializationPct: number;
  ceLoss: number;
  auxLoss: number;
  effectiveExperts: number;
  loads: number[];
}

export interface ProbeRoutingResult {
  sentence: string;
  tokens: Array<{
    text: string;
    topExperts: [number, number];
    weights: [number, number];
    gateProbs: number[];
  }>;
}

export interface MoeEmergenceUpdateMessage {
  type: 'MOE_EMERGENCE_UPDATE';
  step: number;
  seed: number;
  running: boolean;
  auxLossEnabled: boolean;
  auxAlpha: number;
  enabledDomains: Record<DomainId, boolean>;
  vocabSize: number;
  totalParams: number;
  ceLoss: number;
  auxLoss: number;
  effectiveExperts: number;
  specializationPct: number;
  experts: ExpertEmergenceSnapshot[];
  history: EmergenceHistoryPoint[];
  probe: ProbeRoutingResult;
}

export type MoeEmergenceInMessage =
  | { type: 'INIT' }
  | { type: 'START' }
  | { type: 'PAUSE' }
  | { type: 'STEP_BATCH'; count?: number }
  | { type: 'RESEED'; seed?: number }
  | { type: 'SET_AUX_LOSS'; enabled: boolean }
  | { type: 'SET_DOMAIN'; domain: DomainId; enabled: boolean }
  | { type: 'SET_PROBE'; sentence: string };

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

export class RealEmergentMoEModel {
  private readonly N = 4; // 4 Routed Experts
  private readonly K = 2; // Top-2 routing (MiniMind2)
  private readonly D = 24; // Causal attention-pooled representation dim
  private readonly IN_DIM = 24;
  private readonly H_EXP = 7; // Tight bottleneck per routed expert -> forces division of labor!
  private readonly H_SHARED = 4; // Tight shared expert hidden dim (handles common connectors)
  private readonly SHARED_SCALE = 0.2;
  private readonly CUR_WEIGHT = 0.4;
  private readonly CTX_WEIGHT = 0.6;
  private readonly BASE_LR = 0.032;
  private readonly ROUTER_LR_MULT = 2.6;
  private readonly ROUTER_TEMP = 0.48;
  private readonly AUX_ALPHA = 0.2;

  private wordToId = new Map<string, number>();
  private idToWord: string[] = [];
  private V = 0;
  private allSamples: Sample[] = [];

  private E!: AdamParam;
  private Wgate!: AdamParam;
  private bgate!: AdamParam;
  private W1e: AdamParam[] = [];
  private b1e: AdamParam[] = [];
  private W2e: AdamParam[] = [];
  private b2e: AdamParam[] = [];
  private W1s!: AdamParam;
  private b1s!: AdamParam;
  private W2s!: AdamParam;
  private b2s!: AdamParam;
  private Wout!: AdamParam;
  private bout!: AdamParam;

  public seed = 42;
  public step = 0;
  public totalParams = 0;
  public auxLossEnabled = true;
  public enabledDomains: Record<DomainId, boolean> = {
    geo: true,
    code: true,
    food: true,
    math: true,
  };
  public probeSentence = 'Hà Nội có Phở bò và Python tính Đạo hàm';

  private history: EmergenceHistoryPoint[] = [];
  private rng: () => number = mulberry32(42);
  private cursor = 0;
  private lastCeLoss = 0;
  private lastAuxLoss = 0.15;

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

    const byDomain: Record<DomainId, RawSentence[]> = { geo: [], code: [], food: [], math: [] };
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
          const curId = ids[i];
          const contextIds = ids.slice(0, i); // Strictly preceding real words (no cross-domain <bos> dilution)
          const target = i + 1 < ids.length ? ids[i + 1] : 1; // <eos>
          this.allSamples.push({
            curId,
            contextIds,
            target,
            domain: dom,
            tokenText: words[i],
          });
        }
      }
    }
  }

  private randn(): number {
    const u1 = Math.max(1e-7, this.rng());
    const u2 = this.rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  private createParam(size: number, scale: number): AdamParam {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = this.randn() * scale;
    }
    return {
      data,
      grad: new Float32Array(size),
      m: new Float32Array(size),
      v: new Float32Array(size),
    };
  }

  public reseed(newSeed: number): void {
    this.seed = newSeed;
    this.rng = mulberry32(newSeed);
    this.step = 0;
    this.cursor = 0;
    this.history = [];

    this.E = this.createParam(this.V * this.D, 0.24);
    this.Wgate = this.createParam(this.N * this.IN_DIM, 0.18);
    this.bgate = this.createParam(this.N, 0.01);

    this.W1e = [];
    this.b1e = [];
    this.W2e = [];
    this.b2e = [];
    for (let e = 0; e < this.N; e++) {
      this.W1e.push(this.createParam(this.H_EXP * this.IN_DIM, Math.sqrt(1.2 / this.IN_DIM)));
      this.b1e.push(this.createParam(this.H_EXP, 0.02));
      this.W2e.push(this.createParam(this.IN_DIM * this.H_EXP, Math.sqrt(1.2 / this.H_EXP)));
      this.b2e.push(this.createParam(this.IN_DIM, 0.01));
    }

    this.W1s = this.createParam(this.H_SHARED * this.IN_DIM, Math.sqrt(0.8 / this.IN_DIM));
    this.b1s = this.createParam(this.H_SHARED, 0.01);
    this.W2s = this.createParam(this.IN_DIM * this.H_SHARED, Math.sqrt(0.8 / this.H_SHARED));
    this.b2s = this.createParam(this.IN_DIM, 0.01);

    this.Wout = this.createParam(this.V * this.IN_DIM, Math.sqrt(1.0 / this.IN_DIM));
    this.bout = this.createParam(this.V, 0.01);

    this.totalParams =
      this.E.data.length +
      this.Wgate.data.length +
      this.bgate.data.length +
      this.W1e.reduce((s, p) => s + p.data.length, 0) +
      this.b1e.reduce((s, p) => s + p.data.length, 0) +
      this.W2e.reduce((s, p) => s + p.data.length, 0) +
      this.b2e.reduce((s, p) => s + p.data.length, 0) +
      this.W1s.data.length +
      this.b1s.data.length +
      this.W2s.data.length +
      this.b2s.data.length +
      this.Wout.data.length +
      this.bout.data.length;

    this.lastCeLoss = Number(Math.log(this.V).toFixed(4));
    this.lastAuxLoss = this.auxLossEnabled ? this.AUX_ALPHA : 0;
  }

  private getActiveSamples(): Sample[] {
    const active = this.allSamples.filter((s) => this.enabledDomains[s.domain]);
    return active.length > 0 ? active : this.allSamples;
  }

  /**
   * Causal self-attention-like context pooling before the MoE FFN:
   * - If contextIds is empty (first word), returns 1.0 * E[curId] directly (no <bos> sink dilution).
   * - Otherwise x = 0.40 * E[w_t] + 0.60 * mean(E[w_0 .. w_{t-1}])
   */
  private buildContextVec(curId: number, contextIds: number[]): Float32Array {
    const x = new Float32Array(this.IN_DIM);
    const curOff = curId * this.D;
    if (contextIds.length === 0) {
      for (let d = 0; d < this.D; d++) {
        x[d] = this.E.data[curOff + d];
      }
      return x;
    }
    for (let d = 0; d < this.D; d++) {
      x[d] = this.CUR_WEIGHT * this.E.data[curOff + d];
    }
    const ctxScale = this.CTX_WEIGHT / contextIds.length;
    for (let c = 0; c < contextIds.length; c++) {
      const off = contextIds[c] * this.D;
      for (let d = 0; d < this.D; d++) {
        x[d] += ctxScale * this.E.data[off + d];
      }
    }
    return x;
  }

  private routeInput(x: Float32Array): {
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
      const off = e * this.IN_DIM;
      for (let j = 0; j < this.IN_DIM; j++) {
        s += this.Wgate.data[off + j] * x[j];
      }
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

  private forwardMlp(
    x: Float32Array,
    W1: Float32Array,
    b1: Float32Array,
    W2: Float32Array,
    b2: Float32Array,
    hDim: number
  ): { hidden: Float32Array; out: Float32Array } {
    const hidden = new Float32Array(hDim);
    for (let i = 0; i < hDim; i++) {
      let sum = b1[i];
      const off = i * this.IN_DIM;
      for (let j = 0; j < this.IN_DIM; j++) sum += W1[off + j] * x[j];
      hidden[i] = Math.tanh(sum);
    }

    const out = new Float32Array(this.IN_DIM);
    for (let i = 0; i < this.IN_DIM; i++) {
      let sum = b2[i];
      const off = i * hDim;
      for (let j = 0; j < hDim; j++) sum += W2[off + j] * hidden[j];
      out[i] = sum;
    }
    return { hidden, out };
  }

  private allParamsList(): AdamParam[] {
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

  private adamStep(p: AdamParam, t: number, lr: number): void {
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

  public trainOneStep(batchSize = 64): void {
    const pool = this.getActiveSamples();
    for (const p of this.allParamsList()) p.grad.fill(0);

    const invB = 1 / batchSize;
    const counts = [0, 0, 0, 0];
    const sumProbs = [0, 0, 0, 0];

    const batchCaches: Array<{
      x: Float32Array;
      gateProbs: number[];
      dP_ce: [number, number, number, number];
    }> = [];

    let totalCe = 0;
    const dh = new Float32Array(this.IN_DIM);
    const dx = new Float32Array(this.IN_DIM);

    for (let b = 0; b < batchSize; b++) {
      const sample = pool[this.cursor % pool.length];
      this.cursor = (this.cursor + 1) % pool.length;

      const x = this.buildContextVec(sample.curId, sample.contextIds);
      const { gateProbs, topExperts } = this.routeInput(x);
      const [eA, eB] = topExperts;
      // Confidence-preserving Top-2 scaling (unnormalized raw softmax probs p_eA, p_eB)
      // so task gradients wrt probs do NOT cancel out and inactive experts lose probability mass!
      const gA = gateProbs[eA];
      const gB = gateProbs[eB];

      counts[eA] += 1;
      counts[eB] += 1;
      for (let e = 0; e < this.N; e++) sumProbs[e] += gateProbs[e];

      const expA = this.forwardMlp(
        x,
        this.W1e[eA].data,
        this.b1e[eA].data,
        this.W2e[eA].data,
        this.b2e[eA].data,
        this.H_EXP
      );
      const expB = this.forwardMlp(
        x,
        this.W1e[eB].data,
        this.b1e[eB].data,
        this.W2e[eB].data,
        this.b2e[eB].data,
        this.H_EXP
      );
      const expS = this.forwardMlp(
        x,
        this.W1s.data,
        this.b1s.data,
        this.W2s.data,
        this.b2s.data,
        this.H_SHARED
      );

      // Genuine MoE bottleneck: no direct residual bypass x -> h!
      const h = new Float32Array(this.IN_DIM);
      for (let i = 0; i < this.IN_DIM; i++) {
        h[i] = gA * expA.out[i] + gB * expB.out[i] + this.SHARED_SCALE * expS.out[i];
      }

      const logits = new Float32Array(this.V);
      let maxLogit = -Infinity;
      for (let v = 0; v < this.V; v++) {
        let sum = this.bout.data[v];
        const off = v * this.IN_DIM;
        for (let i = 0; i < this.IN_DIM; i++) sum += this.Wout.data[off + i] * h[i];
        logits[v] = sum;
        if (sum > maxLogit) maxLogit = sum;
      }
      const probs = new Float32Array(this.V);
      let sumExp = 0;
      for (let v = 0; v < this.V; v++) {
        const ex = Math.exp(logits[v] - maxLogit);
        probs[v] = ex;
        sumExp += ex;
      }
      const invSum = 1 / sumExp;
      for (let v = 0; v < this.V; v++) probs[v] *= invSum;

      totalCe += -Math.log(Math.max(1e-7, probs[sample.target]));

      dh.fill(0);
      for (let v = 0; v < this.V; v++) {
        const dz = (probs[v] - (v === sample.target ? 1 : 0)) * invB;
        this.bout.grad[v] += dz;
        const off = v * this.IN_DIM;
        for (let i = 0; i < this.IN_DIM; i++) {
          this.Wout.grad[off + i] += dz * h[i];
          dh[i] += dz * this.Wout.data[off + i];
        }
      }

      // Exact task gradient wrt unnormalized gate probs p_eA, p_eB: dL/dp_e = dh · o_e
      let dotA = 0;
      let dotB = 0;
      for (let i = 0; i < this.IN_DIM; i++) {
        dotA += dh[i] * expA.out[i];
        dotB += dh[i] * expB.out[i];
      }
      const dP_ce: [number, number, number, number] = [0, 0, 0, 0];
      dP_ce[eA] = dotA;
      dP_ce[eB] = dotB;
      batchCaches.push({ x, gateProbs, dP_ce });

      // Backprop through Top-2 Routed Experts + Shared Expert -> dx
      dx.fill(0);
      this.backpropMlp(x, expA.hidden, dh, gA, this.W1e[eA], this.b1e[eA], this.W2e[eA], this.b2e[eA], this.H_EXP, dx);
      this.backpropMlp(x, expB.hidden, dh, gB, this.W1e[eB], this.b1e[eB], this.W2e[eB], this.b2e[eB], this.H_EXP, dx);
      this.backpropMlp(x, expS.hidden, dh, this.SHARED_SCALE, this.W1s, this.b1s, this.W2s, this.b2s, this.H_SHARED, dx);

      // Backprop causal context pooling into E[curId] and E[contextIds]
      const curOff = sample.curId * this.D;
      if (sample.contextIds.length === 0) {
        for (let d = 0; d < this.D; d++) {
          this.E.grad[curOff + d] += dx[d];
        }
      } else {
        for (let d = 0; d < this.D; d++) {
          this.E.grad[curOff + d] += this.CUR_WEIGHT * dx[d];
        }
        const ctxScale = this.CTX_WEIGHT / sample.contextIds.length;
        for (let c = 0; c < sample.contextIds.length; c++) {
          const off = sample.contextIds[c] * this.D;
          for (let d = 0; d < this.D; d++) {
            this.E.grad[off + d] += ctxScale * dx[d];
          }
        }
      }
    }

    // MiniMind2 seq_aux Load-Balancing Loss: aux = alpha * sum_e (ce_e * meanP_e)
    const denom = batchSize * (this.K / this.N);
    const ce = counts.map((c) => c / denom);
    const meanP = sumProbs.map((s) => s * invB);
    const alpha = this.auxLossEnabled ? this.AUX_ALPHA : 0;
    const auxLoss = alpha * ce.reduce((acc, c, e) => acc + c * meanP[e], 0);

    const invT = 1 / this.ROUTER_TEMP;
    for (const item of batchCaches) {
      const dP = [0, 0, 0, 0];
      for (let e = 0; e < this.N; e++) {
        dP[e] = item.dP_ce[e] + alpha * ce[e] * invB;
      }
      let dot = 0;
      for (let e = 0; e < this.N; e++) dot += item.gateProbs[e] * dP[e];
      for (let e = 0; e < this.N; e++) {
        const ds = item.gateProbs[e] * (dP[e] - dot) * invT;
        this.bgate.grad[e] += ds;
        const off = e * this.IN_DIM;
        for (let j = 0; j < this.IN_DIM; j++) {
          this.Wgate.grad[off + j] += ds * item.x[j];
        }
      }
    }

    this.step += 1;
    const lr = this.BASE_LR * (0.25 + 0.75 * Math.exp(-this.step / 180));
    for (const p of this.allParamsList()) {
      const paramLr = p === this.Wgate || p === this.bgate ? lr * this.ROUTER_LR_MULT : lr;
      this.adamStep(p, this.step, paramLr);
    }

    this.lastCeLoss = Number((totalCe * invB).toFixed(4));
    this.lastAuxLoss = Number(auxLoss.toFixed(4));
  }

  private backpropMlp(
    x: Float32Array,
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
    for (let i = 0; i < this.IN_DIM; i++) {
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
      const off = j * this.IN_DIM;
      for (let i = 0; i < this.IN_DIM; i++) {
        W1.grad[off + i] += dz1 * x[i];
        dxOut[i] += dz1 * W1.data[off + i];
      }
    }
  }

  public buildSnapshot(running: boolean): MoeEmergenceUpdateMessage {
    const pool = this.getActiveSamples();
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

    for (const sample of pool) {
      const x = this.buildContextVec(sample.curId, sample.contextIds);
      const { topExperts, weights } = this.routeInput(x);
      for (let slot = 0; slot < 2; slot++) {
        const e = topExperts[slot];
        const w = weights[slot];
        expertSlotTotals[e] += w;
        domainCounts[e][sample.domain] += w;

        const key = `${sample.tokenText}|${sample.domain}`;
        const prev = tokenStats[e].get(key);
        if (prev) {
          prev.weightSum += w;
          prev.count += 1;
        } else {
          tokenStats[e].set(key, {
            text: sample.tokenText,
            domain: sample.domain,
            weightSum: w,
            count: 1,
          });
        }
      }
    }

    const grandTotal = expertSlotTotals.reduce((a, b) => a + b, 0) || 1;
    const loads = expertSlotTotals.map((t) => t / grandTotal);
    let entropy = 0;
    for (const l of loads) {
      if (l > 1e-9) entropy -= l * Math.log(l);
    }
    const effectiveExperts = Number(Math.exp(entropy).toFixed(2));

    const enabledCount = DOMAIN_ORDER.filter((d) => this.enabledDomains[d]).length || 1;
    const baselinePurity = 1 / enabledCount;

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
        .slice(0, 6)
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

    const rawSpec =
      enabledCount <= 1
        ? 0
        : Math.max(0, Math.min(1, (weightedPurity - baselinePurity) / (1 - baselinePurity)));
    const specializationPct = Math.round(rawSpec * 100);

    const lastHist = this.history[this.history.length - 1];
    if (!lastHist || lastHist.step !== this.step) {
      this.history.push({
        step: this.step,
        specializationPct,
        ceLoss: this.lastCeLoss,
        auxLoss: this.lastAuxLoss,
        effectiveExperts,
        loads: loads.map((l) => Number(l.toFixed(3))),
      });
      if (this.history.length > 60) this.history.shift();
    }

    return {
      type: 'MOE_EMERGENCE_UPDATE',
      step: this.step,
      seed: this.seed,
      running,
      auxLossEnabled: this.auxLossEnabled,
      auxAlpha: this.auxLossEnabled ? this.AUX_ALPHA : 0,
      enabledDomains: { ...this.enabledDomains },
      vocabSize: this.V,
      totalParams: this.totalParams,
      ceLoss: this.lastCeLoss,
      auxLoss: this.lastAuxLoss,
      effectiveExperts,
      specializationPct,
      experts,
      history: this.history,
      probe: this.probeSentenceRouting(this.probeSentence),
    };
  }

  private probeSentenceRouting(sentence: string): ProbeRoutingResult {
    const words = this.tokenize(sentence);
    const ids = words.map((w) => this.wordToId.get(w) ?? 2);
    const tokens = words.map((text, i) => {
      const curId = ids[i];
      const contextIds = ids.slice(0, i);
      const x = this.buildContextVec(curId, contextIds);
      const { topExperts, weights, gateProbs } = this.routeInput(x);
      return {
        text,
        topExperts,
        weights: [Number(weights[0].toFixed(2)), Number(weights[1].toFixed(2))] as [number, number],
        gateProbs: gateProbs.map((p) => Number(p.toFixed(2))),
      };
    });
    return { sentence, tokens };
  }
}
