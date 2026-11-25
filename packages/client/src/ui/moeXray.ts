import { TelemetryFrame, TokenLayerRouting } from '@minimind-viz/shared';

import { EXPERT_COLORS, SHARED_EXPERT_COLOR as SHARED_COLOR } from '../expertColors';
import {
  DOMAIN_META,
  DOMAIN_ORDER,
  DomainId,
  MoeEmergenceUpdateMessage,
} from '../workers/moeEmergenceCore';
const IDLE_COLOR = '#475569';
const SVG_NS = 'http://www.w3.org/2000/svg';

export interface XrayFocus {
  layerIdx: number;
  tokenIdx: number;
}

export type XrayTab = 'route' | 'balance' | 'emerge';

export interface XrayCallbacks {
  /** Called whenever the focused (layer, token) changes, or with null when the panel closes. */
  onFocusChange(focus: XrayFocus | null): void;
  onSetAuxLoss(enabled: boolean): void;
  onSpeed3x(): void;
  onStateChange?(open: boolean, tab: XrayTab): void;
}

// SVG layout (viewBox 0 0 600 270)
const P_IN = { x: 46, y: 150 };
const P_ROUTER = { x: 150, y: 104 };
const EXP_X = 330;
const EXP_W = 92;
const EXP_H = 34;
const EXP_Y = [32, 82, 132, 182];
const P_SHARED = { x: 330, y: 238 };
const P_SIGMA = { x: 492, y: 150 };
const P_OUT = { x: 566, y: 150 };
const EXP_L = EXP_X - EXP_W / 2;
const EXP_R = EXP_X + EXP_W / 2;

const AUTO_CYCLE_MS = 2600;
const PARTICLE_PERIOD_MS = 1900;
const HISTORY_STEPS = 40;

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
  parent?: Element
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  parent?.appendChild(el);
  return el;
}

function routerToExpertD(e: number): string {
  return `M ${P_ROUTER.x + 22} ${P_ROUTER.y} C 232 ${P_ROUTER.y}, 226 ${EXP_Y[e]}, ${EXP_L} ${EXP_Y[e]}`;
}
function toSigmaD(y: number): string {
  return `M ${EXP_R} ${y} C 432 ${y}, 440 ${P_SIGMA.y}, ${P_SIGMA.x - 20} ${P_SIGMA.y}`;
}
const IN_TO_SHARED_D = `M ${P_IN.x + 14} ${P_IN.y + 12} C 120 ${P_SHARED.y}, 200 ${P_SHARED.y}, ${EXP_L} ${P_SHARED.y}`;

/** Full journey of one copy of the token: x → router → expert → Σ → y. */
function expertRouteD(e: number): { full: string; untilExpertExit: string } {
  const head = `M ${P_IN.x} ${P_IN.y} L ${P_ROUTER.x} ${P_ROUTER.y} C 232 ${P_ROUTER.y}, 226 ${EXP_Y[e]}, ${EXP_L} ${EXP_Y[e]} L ${EXP_R} ${EXP_Y[e]}`;
  return {
    untilExpertExit: head,
    full: `${head} C 432 ${EXP_Y[e]}, 440 ${P_SIGMA.y}, ${P_SIGMA.x} ${P_SIGMA.y} L ${P_OUT.x} ${P_OUT.y}`,
  };
}
const SHARED_ROUTE = {
  untilExpertExit: `M ${P_IN.x} ${P_IN.y} C 120 ${P_SHARED.y}, 200 ${P_SHARED.y}, ${EXP_L} ${P_SHARED.y} L ${EXP_R} ${P_SHARED.y}`,
  full: `M ${P_IN.x} ${P_IN.y} C 120 ${P_SHARED.y}, 200 ${P_SHARED.y}, ${EXP_L} ${P_SHARED.y} L ${EXP_R} ${P_SHARED.y} C 432 ${P_SHARED.y}, 440 ${P_SIGMA.y}, ${P_SIGMA.x} ${P_SIGMA.y} L ${P_OUT.x} ${P_OUT.y}`,
};

interface ParticleRoute {
  path: SVGPathElement;
  total: number;
  exitLen: number;
  color: string;
  weight: number; // radius after the expert = output scaled by gate weight
}

export class MoeXrayPanel {
  private readonly cb: XrayCallbacks;
  private readonly panel = document.getElementById('xray-panel')!;
  private readonly svg = document.getElementById('xray-svg') as unknown as SVGSVGElement;
  private readonly tokensEl = document.getElementById('xray-tokens')!;
  private readonly layersEl = document.getElementById('xray-layers')!;
  private readonly chart = document.getElementById('xray-load-chart') as HTMLCanvasElement;

  private openState = false;
  private tab: 'route' | 'balance' | 'emerge' = 'route';
  private isSimple = true;
  private layerIdx = 0;
  private hoverIdx: number | null = null;
  private pinnedIdx: number | null = null;
  private autoIdx = 1;
  private autoTimer = 0;
  private frame: TelemetryFrame | null = null;
  private tokenSig = '';
  private lastFocusKey = '';
  private lastRouteKey = '';
  private history: Array<{ step: number; loads: number[][] }> = [];

  // Tab 3: Real MoE Emergence Web Worker state
  private emergeWorker: Worker | null = null;
  private lastEmergeSnap: MoeEmergenceUpdateMessage | null = null;
  private emergeCardsBuilt = false;

  // SVG handles
  private rtoE: SVGPathElement[] = [];
  private etoS: SVGPathElement[] = [];
  private expRect: SVGRectElement[] = [];
  private expName: SVGTextElement[] = [];
  private expStatus: SVGTextElement[] = [];
  private probLabel: SVGTextElement[] = [];
  private weightLabel: SVGTextElement[] = [];
  private tokenLabel!: SVGTextElement;
  private particleLayer!: SVGGElement;
  private routes: ParticleRoute[] = [];
  private particles: SVGCircleElement[] = [];
  private rafId = 0;

  constructor(cb: XrayCallbacks) {
    this.cb = cb;
    this.buildSvg();
    this.bindEvents();
  }

  public isOpen(): boolean {
    return this.openState;
  }

  public getTab(): XrayTab {
    return this.tab;
  }

  private syncBgStatusBadge(): void {
    const el = document.getElementById('xray-bg-status');
    if (!el) return;
    if (this.tab === 'balance') {
      el.textContent = '📊 3D nền ẩn · Đang chạy Aux Loss';
      el.classList.add('active-sim');
    } else {
      el.textContent = '⏸ 3D nền tạm dừng';
      el.classList.remove('active-sim');
    }
  }

  private resetActiveSectionScroll(): void {
    const secId =
      this.tab === 'balance' ? 'xray-balance' : this.tab === 'emerge' ? 'xray-emerge' : 'xray-route';
    const sec = document.getElementById(secId);
    if (sec) sec.scrollTop = 0;
  }

  public open(layerIdx?: number): void {
    if (typeof layerIdx === 'number') this.layerIdx = layerIdx;
    this.openState = true;
    this.panel.classList.remove('hidden');
    this.lastFocusKey = '';
    this.lastRouteKey = '';
    this.syncBgStatusBadge();
    this.resetActiveSectionScroll();
    window.clearInterval(this.autoTimer);
    this.autoTimer = window.setInterval(() => this.advanceAuto(), AUTO_CYCLE_MS);
    if (!this.rafId) this.rafId = requestAnimationFrame((t) => this.animateParticles(t));
    if (this.frame) this.update(this.frame);
    this.cb.onStateChange?.(true, this.tab);
  }

  public close(): void {
    if (!this.openState) return;
    this.openState = false;
    this.panel.classList.add('hidden');
    window.clearInterval(this.autoTimer);
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    this.lastFocusKey = '';
    this.pauseEmergeWorker();
    this.cb.onFocusChange(null);
    this.cb.onStateChange?.(false, this.tab);
  }

  public setLevel(isSimple: boolean): void {
    this.isSimple = isSimple;
    this.lastRouteKey = '';
    if (this.frame) this.update(this.frame);
    if (this.lastEmergeSnap) this.renderEmergeNote(this.lastEmergeSnap);
  }

  /** Feed every telemetry frame (before any micro-LM overlay, so `step` is the simulator's). */
  public update(frame: TelemetryFrame): void {
    this.frame = frame;
    this.recordHistory(frame);
    if (!this.openState || frame.architecture !== 'moe') return;

    this.syncTokenChips(frame);
    const focusIdx = this.focusIdx(frame);
    const focusKey = `${this.layerIdx}|${focusIdx}`;
    if (focusKey !== this.lastFocusKey) {
      this.lastFocusKey = focusKey;
      this.cb.onFocusChange({ layerIdx: this.layerIdx, tokenIdx: focusIdx });
    }

    document.getElementById('xray-layer-label')!.textContent = `L${this.layerIdx}`;
    if (this.tab === 'route') this.renderRoute(frame, focusIdx);
    else if (this.tab === 'balance') this.renderBalance(frame);
  }

  // ---------------------------------------------------------------- internals

  private focusIdx(frame: TelemetryFrame): number {
    const n = frame.currentTokens.length;
    const idx = this.hoverIdx ?? this.pinnedIdx ?? this.autoIdx;
    return Math.max(0, Math.min(n - 1, idx));
  }

  private advanceAuto(): void {
    if (!this.frame || this.hoverIdx !== null || this.pinnedIdx !== null || this.tab !== 'route') return;
    this.autoIdx = (this.autoIdx + 1) % Math.max(1, this.frame.currentTokens.length);
    this.update(this.frame);
  }

  private bindEvents(): void {
    document.getElementById('btn-xray-close')!.addEventListener('click', () => this.close());
    document.querySelectorAll<HTMLButtonElement>('#xray-tabs button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const nextTab = btn.dataset.xtab;
        this.tab = nextTab === 'balance' ? 'balance' : nextTab === 'emerge' ? 'emerge' : 'route';
        document
          .querySelectorAll<HTMLButtonElement>('#xray-tabs button')
          .forEach((b) => b.classList.toggle('active', b === btn));
        document.getElementById('xray-route')!.classList.toggle('hidden', this.tab !== 'route');
        document.getElementById('xray-balance')!.classList.toggle('hidden', this.tab !== 'balance');
        document.getElementById('xray-emerge')!.classList.toggle('hidden', this.tab !== 'emerge');
        this.lastRouteKey = '';
        this.syncBgStatusBadge();
        this.resetActiveSectionScroll();
        if (this.tab === 'emerge') {
          this.ensureEmergeWorker();
        } else {
          this.pauseEmergeWorker();
        }
        if (this.frame) this.update(this.frame);
        this.cb.onStateChange?.(this.openState, this.tab);
      });
    });
    document.getElementById('btn-aux-toggle')!.addEventListener('click', () => {
      const enabled = this.frame?.moeBalance?.auxLossEnabled ?? true;
      this.cb.onSetAuxLoss(!enabled);
    });
    document.getElementById('btn-xray-fast')!.addEventListener('click', () => this.cb.onSpeed3x());

    this.bindEmergeControls();

    // Layer strip (built once)
    this.layersEl.innerHTML = Array.from(
      { length: 8 },
      (_, l) => `<button class="xray-layer-btn" data-layer="${l}"><b>L${l}</b><span class="pair"></span></button>`
    ).join('');
    this.layersEl.querySelectorAll<HTMLButtonElement>('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.layerIdx = Number(btn.dataset.layer) || 0;
        if (this.frame) this.update(this.frame);
      });
    });
  }

  private syncTokenChips(frame: TelemetryFrame): void {
    const sig = frame.currentTokens.map((t) => t.text).join('|');
    if (sig !== this.tokenSig) {
      this.tokenSig = sig;
      this.hoverIdx = null;
      this.pinnedIdx = null;
      this.autoIdx = Math.min(1, frame.currentTokens.length - 1);
      this.tokensEl.innerHTML = '';
      frame.currentTokens.forEach((t, idx) => {
        const b = document.createElement('button');
        b.className = 'xray-tok';
        b.textContent = t.text;
        b.addEventListener('mouseenter', () => {
          this.hoverIdx = idx;
          if (this.frame) this.update(this.frame);
        });
        b.addEventListener('mouseleave', () => {
          this.hoverIdx = null;
          if (this.frame) this.update(this.frame);
        });
        b.addEventListener('click', () => {
          this.pinnedIdx = this.pinnedIdx === idx ? null : idx;
          if (this.pinnedIdx === null) this.autoIdx = idx;
          if (this.frame) this.update(this.frame);
        });
        this.tokensEl.appendChild(b);
      });
    }
    const focus = this.focusIdx(frame);
    Array.from(this.tokensEl.children).forEach((el, idx) => {
      el.classList.toggle('focus', idx === focus);
      el.classList.toggle('pinned', idx === this.pinnedIdx);
    });
    document.getElementById('xray-focus-mode')!.textContent =
      this.pinnedIdx !== null ? '📌 Đã ghim — click lần nữa để bỏ' : this.hoverIdx !== null ? '' : '▶ Tự chuyển từ mỗi 2.6 giây';
  }

  private buildSvg(): void {
    const svg = this.svg;
    svg.innerHTML = '';
    const edges = svgEl('g', {}, svg);

    // x -> router, x -> shared, shared -> Σ, Σ -> y
    svgEl('path', { d: `M ${P_IN.x + 18} ${P_IN.y - 8} L ${P_ROUTER.x - 22} ${P_ROUTER.y + 8}`, stroke: '#94a3b8', 'stroke-width': 2.5, fill: 'none' }, edges);
    svgEl('path', { d: IN_TO_SHARED_D, stroke: SHARED_COLOR, 'stroke-width': 3, fill: 'none', opacity: 0.85 }, edges);
    svgEl('path', { d: toSigmaD(P_SHARED.y), stroke: SHARED_COLOR, 'stroke-width': 3, fill: 'none', opacity: 0.85 }, edges);
    svgEl('path', { d: `M ${P_SIGMA.x + 20} ${P_SIGMA.y} L ${P_OUT.x - 15} ${P_OUT.y}`, stroke: '#e2e8f0', 'stroke-width': 3, fill: 'none' }, edges);

    for (let e = 0; e < 4; e++) {
      this.rtoE.push(svgEl('path', { d: routerToExpertD(e), fill: 'none' }, edges));
      this.etoS.push(svgEl('path', { d: toSigmaD(EXP_Y[e]), fill: 'none' }, edges));
    }

    // Nodes
    const nodes = svgEl('g', {}, svg);
    svgEl('circle', { cx: P_IN.x, cy: P_IN.y, r: 18, fill: '#0e7490', stroke: '#22d3ee', 'stroke-width': 2 }, nodes);
    svgEl('text', { x: P_IN.x, y: P_IN.y + 4, 'text-anchor': 'middle', fill: '#fff', 'font-size': 13, 'font-weight': 700 }, nodes).textContent = 'x';
    this.tokenLabel = svgEl('text', { x: P_IN.x, y: P_IN.y - 26, 'text-anchor': 'middle', fill: '#67e8f9', 'font-size': 11, 'font-weight': 700 }, nodes);

    svgEl('circle', { cx: P_ROUTER.x, cy: P_ROUTER.y, r: 22, fill: '#b45309', stroke: '#fbbf24', 'stroke-width': 2 }, nodes);
    svgEl('text', { x: P_ROUTER.x, y: P_ROUTER.y + 4, 'text-anchor': 'middle', fill: '#fff', 'font-size': 10.5, 'font-weight': 700 }, nodes).textContent = 'Router';
    svgEl('text', { x: P_ROUTER.x, y: P_ROUTER.y + 38, 'text-anchor': 'middle', fill: '#fcd34d', 'font-size': 9.5 }, nodes).textContent = 'chọn Top-2';

    for (let e = 0; e < 4; e++) {
      this.expRect.push(
        svgEl('rect', { x: EXP_L, y: EXP_Y[e] - EXP_H / 2, width: EXP_W, height: EXP_H, rx: 7, 'stroke-width': 2 }, nodes)
      );
      this.expName.push(svgEl('text', { x: EXP_X, y: EXP_Y[e] - 2, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 800 }, nodes));
      this.expStatus.push(svgEl('text', { x: EXP_X, y: EXP_Y[e] + 11, 'text-anchor': 'middle', 'font-size': 9.5 }, nodes));
      this.probLabel.push(svgEl('text', { x: 252, y: EXP_Y[e] - 7, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700 }, nodes));
      this.weightLabel.push(svgEl('text', { x: 412, y: EXP_Y[e] - 7, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 800 }, nodes));
    }

    svgEl('rect', { x: EXP_L, y: P_SHARED.y - EXP_H / 2, width: EXP_W, height: EXP_H, rx: 7, fill: 'rgba(16,185,129,0.25)', stroke: SHARED_COLOR, 'stroke-width': 2 }, nodes);
    svgEl('text', { x: EXP_X, y: P_SHARED.y - 2, 'text-anchor': 'middle', fill: '#d1fae5', 'font-size': 11.5, 'font-weight': 800 }, nodes).textContent = 'Shared';
    svgEl('text', { x: EXP_X, y: P_SHARED.y + 11, 'text-anchor': 'middle', fill: '#6ee7b7', 'font-size': 9.5 }, nodes).textContent = 'luôn làm, ×1';
    svgEl('text', { x: 150, y: P_SHARED.y + 26, 'text-anchor': 'middle', fill: '#6ee7b7', 'font-size': 9.5 }, nodes).textContent = 'không qua Router';

    svgEl('circle', { cx: P_SIGMA.x, cy: P_SIGMA.y, r: 20, fill: '#1e293b', stroke: '#e2e8f0', 'stroke-width': 2 }, nodes);
    svgEl('text', { x: P_SIGMA.x, y: P_SIGMA.y + 6, 'text-anchor': 'middle', fill: '#fff', 'font-size': 17, 'font-weight': 800 }, nodes).textContent = 'Σ';
    svgEl('text', { x: P_SIGMA.x, y: P_SIGMA.y + 36, 'text-anchor': 'middle', fill: '#cbd5e1', 'font-size': 9.5 }, nodes).textContent = 'cộng có trọng số';
    svgEl('circle', { cx: P_OUT.x, cy: P_OUT.y, r: 15, fill: '#334155', stroke: '#e2e8f0', 'stroke-width': 2 }, nodes);
    svgEl('text', { x: P_OUT.x, y: P_OUT.y + 5, 'text-anchor': 'middle', fill: '#fff', 'font-size': 13, 'font-weight': 700 }, nodes).textContent = 'y';

    this.particleLayer = svgEl('g', {}, svg);
  }

  private renderRoute(frame: TelemetryFrame, focusIdx: number): void {
    const tok = frame.currentTokens[focusIdx];
    const r: TokenLayerRouting | undefined = tok?.layerRouting[this.layerIdx];
    if (!tok || !r) return;

    const key = `${this.layerIdx}|${focusIdx}|${this.tokenSig}|${JSON.stringify(r)}|${this.isSimple}|${frame.layers[this.layerIdx]?.expertLoads.join()}`;
    if (key === this.lastRouteKey) return;
    this.lastRouteKey = key;

    this.tokenLabel.textContent = `“${tok.text}”`;
    const isSimple = this.isSimple;

    for (let e = 0; e < 4; e++) {
      const slot = r.experts.indexOf(e);
      const active = slot >= 0;
      const w = active ? r.weights[slot] : 0;
      const color = EXPERT_COLORS[e];
      const p = r.gateProbs[e];

      this.rtoE[e].setAttribute('stroke', active ? color : IDLE_COLOR);
      this.rtoE[e].setAttribute('stroke-width', active ? '3' : '1.4');
      this.rtoE[e].setAttribute('stroke-dasharray', active ? '' : '4 4');
      this.rtoE[e].setAttribute('opacity', active ? '0.95' : '0.6');
      // Output edge thickness = gate weight: the copy sent IN is identical, the result coming OUT is scaled.
      this.etoS[e].setAttribute('stroke', active ? color : IDLE_COLOR);
      this.etoS[e].setAttribute('stroke-width', active ? String(1.5 + 9 * w) : '1');
      this.etoS[e].setAttribute('stroke-dasharray', active ? '' : '2 5');
      this.etoS[e].setAttribute('opacity', active ? '0.95' : '0.35');

      this.expRect[e].setAttribute('fill', active ? `${color}33` : 'rgba(30,41,59,0.6)');
      this.expRect[e].setAttribute('stroke', active ? color : IDLE_COLOR);
      this.expName[e].textContent = `E${e}`;
      this.expName[e].setAttribute('fill', active ? '#fff' : '#64748b');
      this.expStatus[e].textContent = active ? (isSimple ? 'đang làm' : 'forward + grad') : '💤 nghỉ (0 phép tính)';
      this.expStatus[e].setAttribute('fill', active ? color : '#64748b');

      this.probLabel[e].textContent = `${Math.round(p * 100)}%`;
      this.probLabel[e].setAttribute('fill', active ? color : '#64748b');
      this.weightLabel[e].textContent = active ? `×${w.toFixed(2)}` : '';
      this.weightLabel[e].setAttribute('fill', color);
    }

    // Particle routes: one per selected expert + the shared expert.
    this.routes.forEach((rt) => rt.path.remove());
    this.routes = [];
    const mk = (d: { full: string; untilExpertExit: string }, color: string, weight: number) => {
      const path = svgEl('path', { d: d.full, fill: 'none', stroke: 'none' }, this.particleLayer);
      const probe = svgEl('path', { d: d.untilExpertExit, fill: 'none', stroke: 'none' }, this.particleLayer);
      const exitLen = probe.getTotalLength();
      probe.remove();
      this.routes.push({ path, total: path.getTotalLength(), exitLen, color, weight });
    };
    r.experts.forEach((e, slot) => mk(expertRouteD(e), EXPERT_COLORS[e], r.weights[slot]));
    mk(SHARED_ROUTE, SHARED_COLOR, 1);
    const need = this.routes.length * 3;
    while (this.particles.length < need) {
      this.particles.push(svgEl('circle', { r: 4 }, this.particleLayer));
    }
    this.particles.forEach((c, i) => c.setAttribute('visibility', i < need ? 'visible' : 'hidden'));

    // Formula + explanation
    const [ea, eb] = r.experts;
    const [wa, wb] = r.weights;
    const span = (e: number, w: number) =>
      `<span style="color:${EXPERT_COLORS[e]};font-weight:800">${w.toFixed(2)}×E${e}(x)</span>`;
    document.getElementById('xray-formula')!.innerHTML =
      `y(“${tok.text}”, L${this.layerIdx}) = ${span(ea, wa)} + ${span(eb, wb)} + <span style="color:${SHARED_COLOR};font-weight:800">Shared(x)</span>` +
      (isSimple ? '' : ' <span style="color:#94a3b8">;  h ← h + y</span>');

    const idle = [0, 1, 2, 3].filter((e) => !r.experts.includes(e));
    const idleTxt = idle.map((e) => `E${e}`).join(', ');
    const probs = r.gateProbs.map((p) => p.toFixed(2)).join(', ');
    document.getElementById('xray-note')!.innerHTML = isSimple
      ? `Router chấm điểm cả 4 chuyên gia cho từ <b>“${tok.text}”</b> rồi chọn 2 người điểm cao nhất: <b style="color:${EXPERT_COLORS[ea]}">E${ea}</b> và <b style="color:${EXPERT_COLORS[eb]}">E${eb}</b>. Từ được gửi <b>nguyên vẹn</b> cho cả hai; kết quả của mỗi người được nhân theo tỉ lệ (${Math.round(wa * 100)}% / ${Math.round(wb * 100)}%) rồi cộng ở Σ, cùng chuyên gia chung (Shared) luôn làm. <b>${idleTxt} nghỉ với riêng từ này</b> — vì vậy mỗi từ chỉ tốn sức ~45M thay vì 145M núm vặn.`
      : `scores = softmax(W_g·x) = [${probs}] → topk(2) = {E${ea}, E${eb}} → chuẩn hóa (norm_topk_prob): ${wa.toFixed(2)} / ${wb.toFixed(2)}. ${idleTxt} không chạy forward cho token này và không nhận gradient từ nó. Router là một lớp Linear riêng của mỗi tầng.`;

    // Same token across all 8 layers
    document.getElementById('xray-layers-caption')!.textContent = isSimple
      ? `Cùng từ “${tok.text}” ở 8 tầng — mỗi tầng có Router riêng nên chọn chuyên gia khác nhau (click để đổi tầng):`
      : `Routing của “${tok.text}” theo tầng (mỗi tầng một gate riêng):`;
    this.layersEl.querySelectorAll<HTMLButtonElement>('button').forEach((btn, l) => {
      btn.classList.toggle('active', l === this.layerIdx);
      const lr = tok.layerRouting[l];
      const pair = btn.querySelector('.pair') as HTMLElement;
      pair.innerHTML = lr
        ? lr.experts.map((e) => `<span style="color:${EXPERT_COLORS[e]}">E${e}</span>`).join('+')
        : '';
    });

    // Whole sentence at this layer: everyone gets work, "idle" is per token only.
    const counts = [0, 0, 0, 0];
    frame.currentTokens.forEach((t) => t.layerRouting[this.layerIdx]?.experts.forEach((e) => (counts[e] += 1)));
    const n = frame.currentTokens.length;
    const jobless = counts.map((c, e) => (c === 0 ? `E${e}` : '')).filter(Boolean);
    const head = isSimple
      ? jobless.length
        ? `⚠ Cả câu ${n} từ ở tầng L${this.layerIdx}: <b>${jobless.join(', ')} không nhận từ nào</b> — việc bị dồn lệch. Xem tab ② Thí nghiệm Aux Loss.`
        : `Cả câu ${n} từ ở tầng L${this.layerIdx} (${n * 2} lượt giao việc): <b>chuyên gia nào cũng có việc</b> — “💤 nghỉ” chỉ đúng với từng từ riêng lẻ.`
      : `Phân bổ Top-2 của câu tại L${this.layerIdx} (${n * 2} lượt):`;
    document.getElementById('xray-sentence')!.innerHTML =
      `<div>${head}</div>` +
      counts
        .map(
          (c, e) => `<div class="load-row">
            <span style="color:${EXPERT_COLORS[e]};font-weight:800">E${e}</span>
            <div class="load-track"><div class="load-fill" style="width:${(c / n) * 100}%;background:${EXPERT_COLORS[e]}"></div></div>
            <span class="mono-text">${c}/${n} từ</span>
          </div>`
        )
        .join('');
  }

  private animateParticles(now: number): void {
    this.rafId = requestAnimationFrame((t) => this.animateParticles(t));
    // The diagram explains routing, not training progress, so it keeps flowing even when training is paused.
    if (!this.openState || this.tab !== 'route') return;
    let i = 0;
    for (const rt of this.routes) {
      for (let k = 0; k < 3; k++) {
        const c = this.particles[i++];
        if (!c) continue;
        const phase = ((now / PARTICLE_PERIOD_MS + k / 3) % 1 + 1) % 1;
        const len = phase * rt.total;
        const pt = rt.path.getPointAtLength(len);
        // Before the expert: an identical copy of x. After it: output scaled by the gate weight.
        const radius = len < rt.exitLen ? 4 : 2 + 5 * rt.weight;
        c.setAttribute('cx', pt.x.toFixed(1));
        c.setAttribute('cy', pt.y.toFixed(1));
        c.setAttribute('r', radius.toFixed(1));
        c.setAttribute('fill', rt.color);
        c.setAttribute('opacity', phase > 0.94 ? String((1 - phase) / 0.06) : '0.95');
      }
    }
  }

  private recordHistory(frame: TelemetryFrame): void {
    if (frame.architecture !== 'moe' || !frame.layers.length) return;
    const last = this.history[this.history.length - 1];
    if (last && frame.step < last.step) this.history = [];
    const tail = this.history[this.history.length - 1];
    if (tail && tail.step === frame.step) return;
    this.history.push({ step: frame.step, loads: frame.layers.map((l) => [...l.expertLoads]) });
    if (this.history.length > HISTORY_STEPS) this.history.shift();
  }

  private renderBalance(frame: TelemetryFrame): void {
    const bal = frame.moeBalance;
    if (!bal) return;
    const isSimple = this.isSimple;
    const layer = frame.layers[this.layerIdx];

    const btn = document.getElementById('btn-aux-toggle') as HTMLButtonElement;
    btn.textContent = bal.auxLossEnabled ? 'Aux Loss: BẬT ✓' : 'Aux Loss: TẮT ✗';
    btn.className = `aux-toggle ${bal.auxLossEnabled ? 'on' : 'off'}`;
    btn.title = bal.auxLossEnabled ? 'Bấm để TẮT luật phạt chia việc lệch' : 'Bấm để BẬT lại luật phạt';

    document.getElementById('xray-balance-stats')!.textContent = isSimple
      ? `Bước ${frame.step} · Số chuyên gia thực sự dùng: ${bal.effectiveExperts.toFixed(1)}/4`
      : `step ${frame.step} · eff. experts ${bal.effectiveExperts.toFixed(2)}/4 · aux(L${this.layerIdx}) ${layer?.auxLoss.toFixed(3)} (α=${bal.auxAlpha})`;

    this.drawLoadChart();

    const loads = layer?.expertLoads ?? [];
    document.getElementById('xray-load-legend')!.innerHTML = loads
      .map((l, e) => `<span><b style="color:${EXPERT_COLORS[e]}">━</b> E${e}: ${Math.round(l * 100)}%</span>`)
      .join('') + `<span style="color:#64748b">Tầng L${this.layerIdx} · ${this.history.length} bước gần nhất</span>`;

    let note: string;
    if (bal.routerFrozen) {
      note = isSimple
        ? '🔒 Đang ở <b>LoRA</b>: Router bị đóng băng (LoRA chỉ gắn vào q_proj/o_proj), nên cách chia việc giữ nguyên — bật/tắt Aux Loss không có tác dụng ở giai đoạn này. Chuyển sang Pretrain hoặc SFT để thử.'
        : '🔒 LoRA: gate không trainable (apply_lora chỉ bọc Linear vuông q_proj/o_proj) → phân bổ tải đứng yên, aux loss không có gradient để tác động.';
    } else if (bal.auxLossEnabled) {
      note = isSimple
        ? '<b>Aux Loss</b> là “luật phạt” cộng thêm vào Loss mỗi khi Router dồn việc lệch cho vài chuyên gia. Đang <b>BẬT</b> → việc được chia khá đều (~25% mỗi người). Thử <b>TẮT</b> rồi xem đồ thị trong vài chục bước (bấm ⏩ cho nhanh).'
        : `MiniMind2 seq_aux: aux = α·Σᵢ ceᵢ·Pᵢ, ceᵢ = countᵢ/(T·k/N), Pᵢ = TB softmax của expert i, α=${bal.auxAlpha}. Chia đều → aux ≈ α; 2/4 expert ôm hết → ≈ 2α. Penalty đẩy router về phân bổ đều.`;
    } else if (bal.collapse < 0.55) {
      note = isSimple
        ? 'Đã <b>TẮT</b> luật phạt. Chuyên gia nào được chọn nhiều hơn sẽ được train nhiều hơn → giỏi hơn → lại càng được chọn nhiều hơn. Sự lệch <b>tăng dần theo từng bước</b>, không xảy ra ngay lập tức.'
        : 'Không có aux loss: gradient chỉ chảy vào expert được chọn → vòng lặp tự củng cố (rich-get-richer). Mất cân bằng tăng dần theo step, tăng tốc khi đã lệch.';
    } else {
      note = isSimple
        ? '⚠ <b>Sụp đổ</b>: 2 chuyên gia ôm gần hết việc (mỗi người ~50% — mức tối đa khi mỗi từ chọn 2 người khác nhau), 2 người còn lại gần như thất nghiệp. GPU vẫn phải chứa đủ 4 chuyên gia nhưng mô hình chỉ thực sự dùng 2. Bật lại Aux Loss để kéo về cân bằng.'
        : '⚠ Load collapse: Top-2 nên mỗi expert tối đa 50% lượt → trạng thái suy biến là 50/50/0/0. Dung lượng hiệu dụng ~2 expert trong khi VRAM vẫn tính cho 4. Ngoài aux loss còn cách khác: DeepSeek-V3 cân bằng không cần aux loss bằng bias động cho từng expert.';
    }
    document.getElementById('xray-balance-note')!.innerHTML = note;
  }

  private drawLoadChart(): void {
    const ctx = this.chart.getContext('2d');
    if (!ctx) return;
    const w = this.chart.width;
    const h = this.chart.height;
    const padL = 34;
    const padR = 10;
    const padT = 10;
    const padB = 18;
    const yMax = 0.6;
    const yOf = (v: number) => padT + (1 - v / yMax) * (h - padT - padB);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '10px Inter, Segoe UI, sans-serif';

    const guide = (v: number, label: string, color: string) => {
      ctx.strokeStyle = color;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, yOf(v));
      ctx.lineTo(w - padR, yOf(v));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.fillText(label, padL + 4, yOf(v) - 3);
    };
    ctx.fillStyle = '#64748b';
    ctx.fillText('0%', 6, yOf(0) + 3);
    ctx.fillText('25%', 4, yOf(0.25) + 3);
    ctx.fillText('50%', 4, yOf(0.5) + 3);
    guide(0.25, 'chia đều (25%)', 'rgba(52,211,153,0.7)');
    guide(0.5, 'trần Top-2 (50%)', 'rgba(251,113,133,0.7)');

    const hist = this.history;
    if (hist.length < 2) {
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('Đang ghi lại tải qua từng bước…', padL + 8, h / 2 + 18);
      return;
    }
    const xOf = (i: number) => padL + (i / (HISTORY_STEPS - 1)) * (w - padL - padR);
    for (let e = 0; e < 4; e++) {
      ctx.strokeStyle = EXPERT_COLORS[e];
      ctx.lineWidth = 2;
      ctx.beginPath();
      hist.forEach((pt, i) => {
        const v = pt.loads[this.layerIdx]?.[e] ?? 0;
        if (i === 0) ctx.moveTo(xOf(i), yOf(v));
        else ctx.lineTo(xOf(i), yOf(v));
      });
      ctx.stroke();
    }
    ctx.fillStyle = '#64748b';
    ctx.fillText(`bước ${hist[0].step}`, padL, h - 4);
    const lastLabel = `bước ${hist[hist.length - 1].step}`;
    ctx.fillText(lastLabel, Math.min(w - padR - ctx.measureText(lastLabel).width, xOf(hist.length - 1) - 20), h - 4);
  }

  // ---------------------------------------------------------------- Tab 3: Real Emergent MoE Lab

  private ensureEmergeWorker(): void {
    if (this.emergeWorker) return;
    const worker = new Worker(new URL('../workers/moeEmergenceWorker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (ev: MessageEvent<MoeEmergenceUpdateMessage>) => {
      if (ev.data?.type === 'MOE_EMERGENCE_UPDATE') {
        this.lastEmergeSnap = ev.data;
        this.renderEmergence(ev.data);
      }
    };
    this.emergeWorker = worker;
    worker.postMessage({ type: 'INIT' });
  }

  private pauseEmergeWorker(): void {
    if (this.emergeWorker) {
      this.emergeWorker.postMessage({ type: 'PAUSE' });
    }
  }

  private bindEmergeControls(): void {
    document.getElementById('btn-emerge-toggle')!.addEventListener('click', () => {
      this.ensureEmergeWorker();
      const isRunning = this.lastEmergeSnap?.running ?? false;
      this.emergeWorker?.postMessage({ type: isRunning ? 'PAUSE' : 'START' });
    });

    document.getElementById('btn-emerge-step')!.addEventListener('click', () => {
      this.ensureEmergeWorker();
      this.emergeWorker?.postMessage({ type: 'STEP_BATCH', count: 15 });
    });

    document.getElementById('btn-emerge-reseed')!.addEventListener('click', () => {
      this.ensureEmergeWorker();
      this.emergeWorker?.postMessage({ type: 'RESEED' });
    });

    document.getElementById('btn-emerge-aux')!.addEventListener('click', () => {
      this.ensureEmergeWorker();
      const curAux = this.lastEmergeSnap?.auxLossEnabled ?? true;
      this.emergeWorker?.postMessage({ type: 'SET_AUX_LOSS', enabled: !curAux });
    });

    document.querySelectorAll<HTMLButtonElement>('#emerge-dom-pills .domain-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.ensureEmergeWorker();
        const dom = btn.dataset.domain as DomainId;
        const enabledMap = this.lastEmergeSnap?.enabledDomains ?? {
          geo: true,
          code: true,
          food: true,
          math: true,
        };
        const currentlyOn = enabledMap[dom];
        const activeCount = DOMAIN_ORDER.filter((d) => enabledMap[d]).length;
        // Keep at least 1 active domain so the training batch never empties
        if (currentlyOn && activeCount <= 1) return;
        this.emergeWorker?.postMessage({ type: 'SET_DOMAIN', domain: dom, enabled: !currentlyOn });
      });
    });

    const probeInput = document.getElementById('emerge-probe-input') as HTMLInputElement;
    const sendProbe = () => {
      this.ensureEmergeWorker();
      this.emergeWorker?.postMessage({ type: 'SET_PROBE', sentence: probeInput.value });
    };
    document.getElementById('btn-emerge-probe')!.addEventListener('click', sendProbe);
    probeInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendProbe();
    });
  }

  private ensureEmergeCardsBuilt(): void {
    if (this.emergeCardsBuilt) return;
    this.emergeCardsBuilt = true;
    const grid = document.getElementById('emerge-grid')!;
    grid.innerHTML = Array.from({ length: 4 }, (_, e) => {
      const segs = DOMAIN_ORDER.map(
        (d) => `<div class="emerge-bar-seg" data-dom="${d}" style="width:25%;background:${DOMAIN_META[d].color}" title="${DOMAIN_META[d].label}"></div>`
      ).join('');
      return `
        <div class="emerge-card" data-expert="${e}">
          <div class="emerge-card-head">
            <span class="emerge-card-title" style="color:${EXPERT_COLORS[e]}">Chuyên gia E${e} (<span class="emerge-load">25%</span>)</span>
            <span class="emerge-card-badge">Chưa phân hóa</span>
          </div>
          <div class="emerge-bar">${segs}</div>
          <div class="emerge-tokens"></div>
        </div>
      `;
    }).join('');
  }

  private renderEmergence(snap: MoeEmergenceUpdateMessage): void {
    this.ensureEmergeCardsBuilt();

    // Update top buttons & stats in place
    const toggleBtn = document.getElementById('btn-emerge-toggle')!;
    toggleBtn.textContent = snap.running ? '⏸ Tạm dừng' : '▶ Bắt đầu Train';
    toggleBtn.classList.toggle('running', snap.running);

    const auxBtn = document.getElementById('btn-emerge-aux')!;
    auxBtn.textContent = snap.auxLossEnabled ? 'Aux Loss thật: BẬT ✓' : 'Aux Loss thật: TẮT ✗';
    auxBtn.className = `aux-toggle ${snap.auxLossEnabled ? 'on' : 'off'}`;

    document.getElementById('emerge-stats')!.textContent =
      `Step ${snap.step} • Seed #${snap.seed} • CE Loss: ${snap.ceLoss.toFixed(3)} • ` +
      `Chuyên gia hiệu dụng: ${snap.effectiveExperts.toFixed(2)}/4 • Phân hóa: ${snap.specializationPct}%`;

    // Sync domain pills
    document.querySelectorAll<HTMLButtonElement>('#emerge-dom-pills .domain-pill').forEach((btn) => {
      const d = btn.dataset.domain as DomainId;
      btn.classList.toggle('active', Boolean(snap.enabledDomains[d]));
    });

    // Update 4 Expert Cards in-place
    const grid = document.getElementById('emerge-grid')!;
    snap.experts.forEach((exp, idx) => {
      const card = grid.children[idx] as HTMLElement;
      (card.querySelector('.emerge-load') as HTMLElement).textContent = `${exp.loadPct}%`;

      const badge = card.querySelector('.emerge-card-badge') as HTMLElement;
      if (exp.loadPct < 3) {
        badge.textContent = '⚠ Bị bỏ đói (0 việc)';
        badge.style.color = '#fda4af';
        badge.style.background = 'rgba(244, 63, 94, 0.2)';
      } else if (snap.step < 6 || !exp.dominantDomain || exp.purity < 0.36) {
        badge.textContent = snap.step === 0 ? 'Ngẫu nhiên (25/25)' : 'Đang tự tìm cụm...';
        badge.style.color = '#cbd5e1';
        badge.style.background = 'rgba(51, 65, 85, 0.7)';
      } else {
        const meta = DOMAIN_META[exp.dominantDomain];
        badge.textContent = `${meta.short} ${Math.round(exp.purity * 100)}%`;
        badge.style.color = meta.color;
        badge.style.background = 'rgba(15, 23, 42, 0.9)';
      }

      DOMAIN_ORDER.forEach((d) => {
        const seg = card.querySelector(`.emerge-bar-seg[data-dom="${d}"]`) as HTMLElement;
        seg.style.width = `${Math.round((exp.domainShares[d] || 0) * 100)}%`;
      });

      const tokBox = card.querySelector('.emerge-tokens') as HTMLElement;
      tokBox.innerHTML = exp.topTokens
        .slice(0, 5)
        .map((t) => {
          const c = DOMAIN_META[t.domain].color;
          return `<span class="emerge-tok-chip" style="border-color:${c}88;background:${c}1a">${t.text}</span>`;
        })
        .join('');
    });

    this.drawEmergenceChart(snap);
    this.renderProbeTokens(snap);
    this.renderEmergeNote(snap);
  }

  private drawEmergenceChart(snap: MoeEmergenceUpdateMessage): void {
    const canvas = document.getElementById('emerge-chart') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const padL = 32;
    const padR = 32;
    const padT = 8;
    const padB = 14;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '9.5px Inter, Segoe UI, sans-serif';

    ctx.strokeStyle = 'rgba(148,163,184,0.16)';
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    const hist = snap.history;
    if (hist.length < 2) {
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('Bấm ▶ Bắt đầu Train hoặc +15 Bước để quan sát sự tự phân hóa qua từng bước...', padL + 6, h / 2 + 3);
      return;
    }

    const xOf = (i: number) => padL + (i / (hist.length - 1)) * (w - padL - padR);
    const ySpec = (pct: number) => padT + (1 - Math.min(100, pct) / 100) * (h - padT - padB);
    const maxLoss = Math.max(2.5, hist[0]?.ceLoss ?? 5.9);
    const yLoss = (l: number) => padT + (1 - Math.min(maxLoss, l) / maxLoss) * (h - padT - padB);

    // 1) CE Loss curve (cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(xOf(i), yLoss(pt.ceLoss));
      else ctx.lineTo(xOf(i), yLoss(pt.ceLoss));
    });
    ctx.stroke();

    // 2) Specialization % curve (emerald)
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(xOf(i), ySpec(pt.specializationPct));
      else ctx.lineTo(xOf(i), ySpec(pt.specializationPct));
    });
    ctx.stroke();

    ctx.fillStyle = '#34d399';
    ctx.fillText(`${hist[hist.length - 1].specializationPct}%`, w - padR + 4, padT + 10);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`${hist[hist.length - 1].ceLoss.toFixed(2)}`, 2, padT + 10);
    ctx.fillStyle = '#64748b';
    ctx.fillText(`Step ${hist[0].step}`, padL, h - 2);
    ctx.fillText(`Step ${hist[hist.length - 1].step}`, w - padR - 42, h - 2);
  }

  private renderProbeTokens(snap: MoeEmergenceUpdateMessage): void {
    const box = document.getElementById('emerge-probe-tokens')!;
    box.innerHTML = snap.probe.tokens
      .map((t) => {
        const [ea, eb] = t.topExperts;
        const [wa, wb] = t.weights;
        return `
          <div class="emerge-probe-tok">
            <b>${t.text.replace(/</g, '&lt;')}</b>
            <span>
              <b style="color:${EXPERT_COLORS[ea]}">E${ea}</b>(${Math.round(wa * 100)}%) +
              <b style="color:${EXPERT_COLORS[eb]}">E${eb}</b>(${Math.round(wb * 100)}%)
            </span>
          </div>
        `;
      })
      .join('');
  }

  private renderEmergeNote(snap: MoeEmergenceUpdateMessage): void {
    const noteEl = document.getElementById('emerge-note')!;
    if (this.isSimple) {
      noteEl.innerHTML =
        `<b>🧠 Không hề có nhãn chủ đề đưa vào mô hình!</b> Ở Step 0, cả 4 chuyên gia đều là số ngẫu nhiên nên chia việc lộn xộn (~25% mỗi nhóm màu). ` +
        `Khi dữ liệu chảy qua và <i>sai số đoán chữ (Backprop)</i> truyền ngược về Router, chuyên gia nào vô tình đoán đúng hơn một chút trên nhóm từ nào sẽ được gửi thêm từ nhóm đó → <b>càng làm càng giỏi</b> (vòng lặp tự củng cố). ` +
        `Bấm <b>🎲 Đổi hạt giống</b> để kiểm chứng vai trò giữa E0..E3 tự đảo khác hẳn nhau!`;
    } else {
      noteEl.innerHTML =
        `<b>🔬 Symmetry Breaking & Co-Adaptation (${snap.totalParams.toLocaleString()} params, Vocab ${snap.vocabSize}):</b> ` +
        `Mạng nhận hoàn toàn ID từ thô qua causal context pooling (<code>0.4·E[w_t] + 0.6·mean(E[w_&lt;t])</code>), rồi đi qua <code>Softmax(W_g·x) → Top-2 → 4 Routed (H=7) + Shared (H=4)</code>. ` +
        `Đạo hàm <code>dL_CE/dp_e = dh·o_e</code> kết hợp với <code>seq_aux (α=${snap.auxAlpha})</code> tự đẩy các từ có ngữ cảnh gần nhau về cùng chuyên gia — màu 4 nhóm chỉ dùng ở tầng Observer để đo độ thuần khiết.`;
    }
  }
}
