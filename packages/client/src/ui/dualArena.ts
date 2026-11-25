import { EXPERT_COLORS, SHARED_EXPERT_COLOR as SHARED_COLOR } from '../expertColors';
import {
  ARENA_GEN_PROMPTS,
  ARENA_PROBE_WORDS,
  DualArenaUpdateMessage,
} from '../workers/dualArenaCore';
import { DOMAIN_META, DOMAIN_ORDER, DomainId } from '../workers/moeEmergenceCore';

const SVG_NS = 'http://www.w3.org/2000/svg';

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

export class DualArenaModal {
  private readonly modal = document.getElementById('arena-modal')!;
  private readonly denseSvg = document.getElementById('arena-dense-svg') as unknown as SVGSVGElement;
  private readonly moeSvg = document.getElementById('arena-moe-svg') as unknown as SVGSVGElement;
  private readonly chart = document.getElementById('arena-chart') as HTMLCanvasElement;

  private worker: Worker | null = null;
  private isOpenState = false;
  private isSimple = true;
  private selectedProbe: DomainId = 'geo';
  private latestMsg: DualArenaUpdateMessage | null = null;

  // Dense SVG handles
  private denseTokText!: SVGTextElement;
  private denseFfnRect!: SVGRectElement;
  private denseFfnLabel!: SVGTextElement;

  // MoE SVG handles
  private moeTokText!: SVGTextElement;
  private moeInPaths: SVGPathElement[] = [];
  private moeOutPaths: SVGPathElement[] = [];
  private moeExpRects: SVGRectElement[] = [];
  private moeExpStatus: SVGTextElement[] = [];

  private builtStaticGrids = false;

  constructor(private readonly onOpenChange?: (open: boolean) => void) {
    this.buildDenseSvg();
    this.buildMoeSvg();
    this.ensureStaticContainersBuilt();
    this.bindEvents();
  }

  public isOpen(): boolean {
    return this.isOpenState;
  }

  public open(): void {
    this.isOpenState = true;
    this.modal.classList.remove('hidden');
    this.ensureWorker();
    if (this.latestMsg) this.render(this.latestMsg);
    this.onOpenChange?.(true);
  }

  public close(): void {
    if (!this.isOpenState) return;
    this.isOpenState = false;
    this.modal.classList.add('hidden');
    this.worker?.postMessage({ type: 'PAUSE' });
    this.onOpenChange?.(false);
  }

  public setLevel(isSimple: boolean): void {
    this.isSimple = isSimple;
    if (this.latestMsg) this.renderSummary(this.latestMsg);
  }

  private ensureWorker(): void {
    if (this.worker) return;
    const w = new Worker(new URL('../workers/dualArenaWorker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = (ev: MessageEvent<DualArenaUpdateMessage>) => {
      if (ev.data?.type === 'DUAL_ARENA_UPDATE') {
        this.latestMsg = ev.data;
        if (this.isOpenState) this.render(ev.data);
      }
    };
    this.worker = w;
    w.postMessage({ type: 'INIT' });
  }

  private bindEvents(): void {
    document.getElementById('btn-arena-close')!.addEventListener('click', () => this.close());
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) this.close();
    });

    document.getElementById('btn-arena-run')!.addEventListener('click', () => {
      this.ensureWorker();
      const running = this.latestMsg?.running ?? false;
      this.worker?.postMessage({ type: running ? 'PAUSE' : 'START' });
    });

    document.getElementById('btn-arena-step')!.addEventListener('click', () => {
      this.ensureWorker();
      this.worker?.postMessage({ type: 'RUN_ONE_EPOCH' });
    });

    document.getElementById('btn-arena-reseed')!.addEventListener('click', () => {
      this.ensureWorker();
      this.worker?.postMessage({ type: 'RESEED' });
    });

    document.getElementById('btn-arena-aux')!.addEventListener('click', () => {
      this.ensureWorker();
      const curAux = this.latestMsg?.auxLossEnabled ?? true;
      this.worker?.postMessage({ type: 'SET_AUX_LOSS', enabled: !curAux });
    });

    const slider = document.getElementById('arena-epoch-slider') as HTMLInputElement;
    slider.addEventListener('input', () => {
      this.ensureWorker();
      this.worker?.postMessage({ type: 'SELECT_EPOCH', epoch: Number(slider.value) });
    });

    document.querySelectorAll<HTMLButtonElement>('#arena-probe-btns button').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.selectedProbe = (btn.dataset.probe as DomainId) || 'geo';
        document
          .querySelectorAll<HTMLButtonElement>('#arena-probe-btns button')
          .forEach((b) => b.classList.toggle('active', b === btn));
        if (this.latestMsg) this.render(this.latestMsg);
      });
    });
  }

  private buildDenseSvg(): void {
    const svg = this.denseSvg;
    svg.innerHTML = '';

    // Pipeline matching actual DenseCore: x -> Context Pooling -> FFN (tanh, 18/18) -> y (no attention, no residual skip)
    svgEl('path', { d: 'M 42 95 L 76 95', stroke: '#94a3b8', 'stroke-width': 2.4, fill: 'none' }, svg);
    svgEl('path', { d: 'M 188 95 L 218 95', stroke: '#38bdf8', 'stroke-width': 2.8, fill: 'none' }, svg);
    svgEl('path', { d: 'M 376 95 L 414 95', stroke: '#818cf8', 'stroke-width': 2.8, fill: 'none' }, svg);

    svgEl('circle', { cx: 28, cy: 95, r: 14, fill: '#0e7490', stroke: '#22d3ee', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 28, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, 'font-weight': 800 }, svg).textContent = 'x';
    this.denseTokText = svgEl('text', { x: 46, y: 62, 'text-anchor': 'middle', fill: '#67e8f9', 'font-size': 11, 'font-weight': 800 }, svg);

    // Causal Context Pooling block (same as MoE)
    svgEl('rect', { x: 76, y: 64, width: 112, height: 62, rx: 7, fill: 'rgba(59,130,246,0.22)', stroke: '#3b82f6', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 132, y: 90, 'text-anchor': 'middle', fill: '#dbeafe', 'font-size': 10.5, 'font-weight': 800 }, svg).textContent = 'Trộn ngữ cảnh';
    svgEl('text', { x: 132, y: 106, 'text-anchor': 'middle', fill: '#93c5fd', 'font-size': 9 }, svg).textContent = '0.4·từ + 0.6·TB';

    // Single monolithic tanh FFN block (18/18 active)
    this.denseFfnRect = svgEl('rect', { x: 218, y: 50, width: 158, height: 90, rx: 9, fill: 'rgba(99,102,241,0.34)', stroke: '#818cf8', 'stroke-width': 2.5 }, svg);
    svgEl('text', { x: 297, y: 82, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11.5, 'font-weight': 800 }, svg).textContent = '1 Khối FFN (tanh)';
    this.denseFfnLabel = svgEl('text', { x: 297, y: 101, 'text-anchor': 'middle', fill: '#c7d2fe', 'font-size': 10 }, svg);
    svgEl('text', { x: 297, y: 120, 'text-anchor': 'middle', fill: '#a5b4fc', 'font-size': 9.5 }, svg).textContent = '100% nơ-ron luôn tính';

    svgEl('circle', { cx: 428, cy: 95, r: 14, fill: '#334155', stroke: '#e2e8f0', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 428, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, 'font-weight': 800 }, svg).textContent = 'y';
  }

  private buildMoeSvg(): void {
    const svg = this.moeSvg;
    svg.innerHTML = '';

    const expY = [24, 62, 100, 138];
    const sharedY = 172;

    svgEl('path', { d: 'M 37 95 L 60 95', stroke: '#94a3b8', 'stroke-width': 2, fill: 'none' }, svg);
    // Context Pooling -> Router
    svgEl('path', { d: 'M 152 95 L 178 80', stroke: '#fbbf24', 'stroke-width': 2.2, fill: 'none' }, svg);
    // Context Pooling -> Shared (no router)
    svgEl('path', { d: `M 152 112 C 168 ${sharedY}, 210 ${sharedY}, 266 ${sharedY}`, stroke: SHARED_COLOR, 'stroke-width': 2.2, fill: 'none' }, svg);
    svgEl('path', { d: `M 344 ${sharedY} C 382 ${sharedY}, 390 95, 404 95`, stroke: SHARED_COLOR, 'stroke-width': 2.2, fill: 'none' }, svg);
    svgEl('path', { d: 'M 426 95 L 438 95', stroke: '#e2e8f0', 'stroke-width': 2.5, fill: 'none' }, svg);

    for (let e = 0; e < 4; e++) {
      this.moeInPaths.push(
        svgEl('path', { d: `M 218 80 C 242 80, 240 ${expY[e]}, 266 ${expY[e]}`, fill: 'none' }, svg)
      );
      this.moeOutPaths.push(
        svgEl('path', { d: `M 344 ${expY[e]} C 380 ${expY[e]}, 386 95, 404 95`, fill: 'none' }, svg)
      );
    }

    svgEl('circle', { cx: 24, cy: 95, r: 13, fill: '#0e7490', stroke: '#22d3ee', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 24, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 10.5, 'font-weight': 800 }, svg).textContent = 'x';
    this.moeTokText = svgEl('text', { x: 42, y: 64, 'text-anchor': 'middle', fill: '#67e8f9', 'font-size': 10.5, 'font-weight': 800 }, svg);

    // Causal Context Pooling block (identical to Dense)
    svgEl('rect', { x: 60, y: 68, width: 92, height: 54, rx: 6, fill: 'rgba(59,130,246,0.22)', stroke: '#3b82f6', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 106, y: 91, 'text-anchor': 'middle', fill: '#dbeafe', 'font-size': 10, 'font-weight': 800 }, svg).textContent = 'Trộn ngữ cảnh';
    svgEl('text', { x: 106, y: 106, 'text-anchor': 'middle', fill: '#93c5fd', 'font-size': 8.4 }, svg).textContent = '0.4·từ + 0.6·TB';

    // Router
    svgEl('circle', { cx: 198, cy: 80, r: 20, fill: '#b45309', stroke: '#fbbf24', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 198, y: 83, 'text-anchor': 'middle', fill: '#fff', 'font-size': 9.5, 'font-weight': 800 }, svg).textContent = 'Router';

    // 4 Routed Experts
    for (let e = 0; e < 4; e++) {
      this.moeExpRects.push(
        svgEl('rect', { x: 266, y: expY[e] - 13, width: 78, height: 26, rx: 6, 'stroke-width': 1.8 }, svg)
      );
      this.moeExpStatus.push(
        svgEl('text', { x: 305, y: expY[e] + 4, 'text-anchor': 'middle', 'font-size': 9.6, 'font-weight': 800 }, svg)
      );
    }

    // Shared Expert (scale ×1.0)
    svgEl('rect', { x: 266, y: sharedY - 12, width: 78, height: 24, rx: 6, fill: 'rgba(16,185,129,0.25)', stroke: SHARED_COLOR, 'stroke-width': 1.8 }, svg);
    svgEl('text', { x: 305, y: sharedY + 4, 'text-anchor': 'middle', fill: '#d1fae5', 'font-size': 9.3, 'font-weight': 800 }, svg).textContent = 'Shared (4) ×1';

    // Sigma + output y
    svgEl('circle', { cx: 414, cy: 95, r: 14, fill: '#1e293b', stroke: '#e2e8f0', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 414, y: 100, 'text-anchor': 'middle', fill: '#fff', 'font-size': 13, 'font-weight': 800 }, svg).textContent = 'Σ';
    svgEl('circle', { cx: 446, cy: 95, r: 11, fill: '#334155', stroke: '#e2e8f0', 'stroke-width': 1.8 }, svg);
    svgEl('text', { x: 446, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 10, 'font-weight': 800 }, svg).textContent = 'y';
  }

  private ensureStaticContainersBuilt(): void {
    if (this.builtStaticGrids) return;
    this.builtStaticGrids = true;

    // 18 Dense heatmap cells
    const heatEl = document.getElementById('arena-dense-heat')!;
    heatEl.innerHTML = Array.from(
      { length: 18 },
      (_, i) => `<div class="arena-heat-cell" data-n="${i}">n${i}</div>`
    ).join('');

    // 4 MoE expert mini cards
    const moeEl = document.getElementById('arena-moe-experts')!;
    moeEl.innerHTML = Array.from({ length: 4 }, (_, e) => {
      const segs = DOMAIN_ORDER.map(
        (d) => `<div class="emerge-bar-seg" data-dom="${d}" style="width:25%;background:${DOMAIN_META[d].color}"></div>`
      ).join('');
      return `
        <div class="arena-moe-card" data-expert="${e}">
          <div class="arena-moe-card-head">
            <span style="color:${EXPERT_COLORS[e]}">E${e} (<span class="m-load">25%</span>)</span>
            <span class="m-dom" style="color:#cbd5e1">Ngẫu nhiên</span>
          </div>
          <div class="emerge-bar">${segs}</div>
        </div>
      `;
    }).join('');

    // 4 Generation cards for both Dense & MoE
    const buildGens = (id: string) => {
      const el = document.getElementById(id)!;
      el.innerHTML = ARENA_GEN_PROMPTS.map(
        (p) => `<div class="arena-gen-item"><b>“${p}” →</b> <span class="gen-txt"></span></div>`
      ).join('');
    };
    buildGens('arena-dense-gen');
    buildGens('arena-moe-gen');
  }

  private render(msg: DualArenaUpdateMessage): void {
    const snap = msg.current;
    const probeRef = ARENA_PROBE_WORDS.find((p) => p.id === this.selectedProbe) ?? ARENA_PROBE_WORDS[0];

    // Toolbar controls
    document.getElementById('arena-milestone')!.textContent = snap.stageLabel;

    const runBtn = document.getElementById('btn-arena-run')!;
    runBtn.textContent = msg.running ? '⏸ Tạm dừng' : '▶ Chạy liên tục';
    runBtn.classList.toggle('running', msg.running);

    const auxBtn = document.getElementById('btn-arena-aux')!;
    auxBtn.textContent = msg.auxLossEnabled ? 'Aux Loss MoE: BẬT ✓' : 'Aux Loss MoE: TẮT ✗';
    auxBtn.className = `aux-toggle ${msg.auxLossEnabled ? 'on' : 'off'}`;

    const slider = document.getElementById('arena-epoch-slider') as HTMLInputElement;
    slider.max = String(msg.latestEpoch);
    slider.value = String(msg.viewedEpoch);
    document.getElementById('arena-epoch-readout')!.textContent =
      `Epoch ${msg.viewedEpoch} / ${msg.latestEpoch}`;

    // ---- LEFT COLUMN: DENSE ----
    const dProbe = snap.dense.probeByDomain[this.selectedProbe];
    document.getElementById('arena-dense-badge')!.textContent =
      `18/18 active • CE Loss: ${snap.dense.ceLoss.toFixed(3)}`;
    this.denseTokText.textContent = `“${probeRef.word}”`;
    this.denseFfnLabel.textContent = `18/18 nơ-ron • “${probeRef.word}”`;
    this.denseFfnRect.setAttribute('fill', `rgba(99,102,241,${0.25 + 0.35 * (1 - Math.min(1, snap.dense.ceLoss / 6))})`);

    const heatEl = document.getElementById('arena-dense-heat')!;
    dProbe.activations.forEach((act, idx) => {
      const cell = heatEl.children[idx] as HTMLElement;
      const alpha = (0.15 + act * 0.78).toFixed(2);
      cell.style.background = `rgba(129, 140, 248, ${alpha})`;
      cell.textContent = act.toFixed(1);
    });

    document.getElementById('arena-dense-pred')!.innerHTML =
      `<span>Top-3 đoán sau “${probeRef.word}”:</span>` +
      dProbe.top3Next
        .map((c) => `<span class="arena-pred-chip">${c.word} ${Math.round(c.prob * 100)}%</span>`)
        .join('');

    const denseGenEl = document.getElementById('arena-dense-gen')!;
    snap.dense.generations.forEach((g, idx) => {
      ((denseGenEl.children[idx] as HTMLElement).querySelector('.gen-txt') as HTMLElement).textContent =
        g.continuation;
    });

    // ---- RIGHT COLUMN: MoE ----
    const mProbe = snap.moe.probeByDomain[this.selectedProbe];
    document.getElementById('arena-moe-badge')!.textContent =
      `18/32 active • CE Loss: ${snap.moe.ceLoss.toFixed(3)} • Eff: ${snap.moe.effectiveExperts.toFixed(2)}/4`;
    this.moeTokText.textContent = `“${probeRef.word}”`;

    for (let e = 0; e < 4; e++) {
      const slot = mProbe.topExperts.indexOf(e);
      const active = slot >= 0;
      const w = active ? mProbe.weights[slot] : 0;
      const color = EXPERT_COLORS[e];

      this.moeInPaths[e].setAttribute('stroke', active ? color : '#475569');
      this.moeInPaths[e].setAttribute('stroke-width', active ? '2.6' : '1.2');
      this.moeInPaths[e].setAttribute('stroke-dasharray', active ? '' : '3 3');
      this.moeInPaths[e].setAttribute('opacity', active ? '0.95' : '0.45');

      this.moeOutPaths[e].setAttribute('stroke', active ? color : '#475569');
      this.moeOutPaths[e].setAttribute('stroke-width', active ? String(1.5 + 6.5 * w) : '1');
      this.moeOutPaths[e].setAttribute('stroke-dasharray', active ? '' : '2 4');
      this.moeOutPaths[e].setAttribute('opacity', active ? '0.95' : '0.3');

      this.moeExpRects[e].setAttribute('fill', active ? `${color}33` : 'rgba(30,41,59,0.6)');
      this.moeExpRects[e].setAttribute('stroke', active ? color : '#475569');
      this.moeExpStatus[e].textContent = active ? `E${e} ⚡ ×${w.toFixed(2)}` : `E${e} 💤 nghỉ`;
      this.moeExpStatus[e].setAttribute('fill', active ? '#fff' : '#64748b');
    }

    const moeGridEl = document.getElementById('arena-moe-experts')!;
    snap.moe.experts.forEach((exp, idx) => {
      const card = moeGridEl.children[idx] as HTMLElement;
      (card.querySelector('.m-load') as HTMLElement).textContent = `${exp.loadPct}%`;
      const domEl = card.querySelector('.m-dom') as HTMLElement;
      if (exp.loadPct < 4) {
        domEl.textContent = '⚠ 0 việc';
        domEl.style.color = '#fda4af';
      } else if (snap.epoch === 0 || !exp.dominantDomain || exp.purity < 0.36) {
        domEl.textContent = 'Ngẫu nhiên';
        domEl.style.color = '#94a3b8';
      } else {
        const meta = DOMAIN_META[exp.dominantDomain];
        domEl.textContent = `${meta.short} ${Math.round(exp.purity * 100)}%`;
        domEl.style.color = meta.color;
      }
      DOMAIN_ORDER.forEach((d) => {
        (card.querySelector(`.emerge-bar-seg[data-dom="${d}"]`) as HTMLElement).style.width =
          `${Math.round((exp.domainShares[d] || 0) * 100)}%`;
      });
    });

    document.getElementById('arena-moe-pred')!.innerHTML =
      `<span>Top-3 đoán sau “${probeRef.word}”:</span>` +
      mProbe.top3Next
        .map((c) => `<span class="arena-pred-chip">${c.word} ${Math.round(c.prob * 100)}%</span>`)
        .join('');

    const moeGenEl = document.getElementById('arena-moe-gen')!;
    snap.moe.generations.forEach((g, idx) => {
      ((moeGenEl.children[idx] as HTMLElement).querySelector('.gen-txt') as HTMLElement).textContent =
        g.continuation;
    });

    this.drawDualChart(msg);
    this.renderSummary(msg);
  }

  private drawDualChart(msg: DualArenaUpdateMessage): void {
    const ctx = this.chart.getContext('2d');
    if (!ctx) return;
    const w = this.chart.width;
    const h = this.chart.height;
    const padL = 44;
    const padR = 44;
    const padT = 16;
    const padB = 18;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '10px Inter, Segoe UI, sans-serif';

    const hist = msg.history;
    if (hist.length < 2) {
      ctx.strokeStyle = 'rgba(148,163,184,0.22)';
      ctx.beginPath();
      ctx.moveTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(
        'Bấm ▶ Chạy liên tục hoặc ⏭ Chạy 1 Epoch rồi dừng để xem CE Loss (Dense nhỏ, MoE, Dense to) & Độ phân hóa...',
        padL + 6,
        h / 2 + 4
      );
      return;
    }

    const maxX = Math.max(1, hist.length - 1);
    const xOf = (i: number) => padL + (i / maxX) * plotW;
    const maxLossRaw = hist.reduce(
      (m, pt) => Math.max(m, pt.denseCeLoss, pt.moeCeLoss, pt.denseBigCeLoss),
      2.5
    );
    const maxLoss = Math.ceil(maxLossRaw * 2) / 2;
    const yLoss = (l: number) => padT + (1 - Math.max(0, Math.min(maxLoss, l)) / maxLoss) * plotH;
    const ySpec = (pct: number) => padT + (1 - Math.max(0, Math.min(100, pct)) / 100) * plotH;

    // Horizontal dashed gridlines at 100%, 50%, 0%
    ctx.strokeStyle = 'rgba(148,163,184,0.16)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (const frac of [0, 0.5, 1]) {
      const y = padT + frac * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Left & right Y-axis spines + bottom axis
    ctx.strokeStyle = 'rgba(148,163,184,0.32)';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.lineTo(w - padR, padT);
    ctx.stroke();

    // Axis titles
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText('CE Loss', 4, 11);
    ctx.fillStyle = '#34d399';
    ctx.textAlign = 'right';
    ctx.fillText('Phân hóa', w - 4, 11);

    // Left Y-axis ticks (Loss)
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.fillText(maxLoss.toFixed(1), padL - 5, padT + 4);
    ctx.fillText((maxLoss / 2).toFixed(1), padL - 5, padT + plotH * 0.5 + 3);
    ctx.fillText('0.0', padL - 5, h - padB + 3);

    // Right Y-axis ticks (Specialization %)
    ctx.fillStyle = '#34d399';
    ctx.textAlign = 'left';
    ctx.fillText('100%', w - padR + 5, padT + 4);
    ctx.fillText('50%', w - padR + 5, padT + plotH * 0.5 + 3);
    ctx.fillText('0%', w - padR + 5, h - padB + 3);

    // Scrubber cursor line
    const cursorX = xOf(msg.viewedEpoch);
    ctx.strokeStyle = 'rgba(254, 240, 138, 0.65)';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cursorX, padT);
    ctx.lineTo(cursorX, h - padB);
    ctx.stroke();
    ctx.setLineDash([]);

    const drawSeries = (
      color: string,
      fn: (pt: (typeof hist)[0]) => number,
      width = 2.1,
      dash: number[] = [],
      alpha = 1
    ) => {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      hist.forEach((pt, i) => {
        const y = fn(pt);
        if (i === 0) ctx.moveTo(xOf(i), y);
        else ctx.lineTo(xOf(i), y);
      });
      ctx.stroke();
      ctx.restore();
    };

    drawSeries('#818cf8', (pt) => yLoss(pt.denseCeLoss), 2.2);
    drawSeries('#f59e0b', (pt) => yLoss(pt.moeCeLoss), 2.2);
    drawSeries('#38bdf8', (pt) => yLoss(pt.denseBigCeLoss), 2.0, [5, 3]);
    drawSeries('#34d399', (pt) => ySpec(pt.moeSpecPct), 1.5, [], 0.85);

    // X-axis epoch ticks
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText('Ep 0', padL, h - 4);
    const lastEp = hist[hist.length - 1].epoch;
    if (lastEp >= 4) {
      const midEp = Math.round(lastEp / 2);
      ctx.textAlign = 'center';
      ctx.fillText(`Ep ${midEp}`, xOf(midEp), h - 4);
    }
    ctx.textAlign = 'right';
    ctx.fillText(`Ep ${lastEp}`, w - padR, h - 4);
    ctx.textAlign = 'left';
  }

  private renderSummary(msg: DualArenaUpdateMessage): void {
    const snap = msg.current;
    const summaryEl = document.getElementById('arena-summary')!;
    const moeTotalRatio = (snap.moe.totalParams / snap.dense.totalParams).toFixed(2);
    const bigTotalRatio = (snap.denseBig.totalParams / snap.dense.totalParams).toFixed(2);
    const bigComputeRatio = (snap.denseBig.activeFfnNeurons / snap.dense.activeFfnNeurons).toFixed(2);
    const moeFfnRatio = (snap.moe.ffnParams / snap.dense.ffnParams).toFixed(2);
    const bigFfnRatio = (snap.denseBig.ffnParams / snap.dense.ffnParams).toFixed(2);
    const nonFfnParams = snap.dense.totalParams - snap.dense.ffnParams;
    const nonFfnMoePct = Math.round((nonFfnParams / snap.moe.totalParams) * 100);

    const fmtN = (n: number) => n.toLocaleString('en-US');
    const auxBadge = snap.auxLossEnabled
      ? `BẬT (Aux=${snap.moe.auxLoss.toFixed(3)})`
      : `TẮT (α=0)`;

    const tableRows = this.isSimple
      ? `
        <tr>
          <td>⚡ Nơ-ron FFN tính / từ</td>
          <td>${snap.dense.activeFfnNeurons} (1×)</td>
          <td class="cur">${snap.moe.activeFfnNeurons} (1×)</td>
          <td class="big">${snap.denseBig.activeFfnNeurons} (${bigComputeRatio}×)</td>
        </tr>
        <tr>
          <td>🧠 Tổng tham số</td>
          <td>${fmtN(snap.dense.totalParams)} (1×)</td>
          <td class="cur">${fmtN(snap.moe.totalParams)} (${moeTotalRatio}×)</td>
          <td class="big">${fmtN(snap.denseBig.totalParams)} (${bigTotalRatio}×)</td>
        </tr>
        <tr>
          <td>📉 CE Loss (Ep ${snap.epoch})</td>
          <td>${snap.dense.ceLoss.toFixed(3)}</td>
          <td class="cur">${snap.moe.ceLoss.toFixed(3)}</td>
          <td class="big">${snap.denseBig.ceLoss.toFixed(3)}</td>
        </tr>`
      : `
        <tr>
          <td>Active / Total FFN</td>
          <td>${snap.dense.activeFfnNeurons}/${snap.dense.totalFfnNeurons} (1×)</td>
          <td class="cur">${snap.moe.activeFfnNeurons}/${snap.moe.totalFfnNeurons} (1×)</td>
          <td class="big">${snap.denseBig.activeFfnNeurons}/${snap.denseBig.totalFfnNeurons} (${bigComputeRatio}×)</td>
        </tr>
        <tr>
          <td>FFN params</td>
          <td>${fmtN(snap.dense.ffnParams)} (1×)</td>
          <td class="cur">${fmtN(snap.moe.ffnParams)} (${moeFfnRatio}×)</td>
          <td class="big">${fmtN(snap.denseBig.ffnParams)} (${bigFfnRatio}×)</td>
        </tr>
        <tr>
          <td>Total params</td>
          <td>${fmtN(snap.dense.totalParams)} (1×)</td>
          <td class="cur">${fmtN(snap.moe.totalParams)} (${moeTotalRatio}×)</td>
          <td class="big">${fmtN(snap.denseBig.totalParams)} (${bigTotalRatio}×)</td>
        </tr>
        <tr>
          <td>CE Loss (Ep ${snap.epoch})</td>
          <td><b>${snap.dense.ceLoss.toFixed(4)}</b></td>
          <td class="cur"><b>${snap.moe.ceLoss.toFixed(4)}</b></td>
          <td class="big"><b>${snap.denseBig.ceLoss.toFixed(4)}</b></td>
        </tr>`;

    const notesHtml = this.isSimple
      ? `<div class="arena-summary-notes">
          • <b>Trạng thái MoE:</b> Aux Loss <b>${auxBadge}</b> · Độ lệch tải <code>ce·P = ${snap.moe.rawImbalance.toFixed(2)}</code> (1.00 = đều) · Chuyên gia dùng thật: <b>${snap.moe.effectiveExperts.toFixed(2)}/4</b> · Phân hóa: <b>${snap.moe.specializationPct}%</b>.<br/>
          • <b>Kết quả thực tế trên 64 câu (${msg.totalSamples} token/epoch):</b> cùng tính 18 nơ-ron/từ, <b>MoE và Dense nhỏ đạt Loss gần ngang nhau</b>; <b>Dense to (32)</b> học sát hơn vì tính gấp ${bigComputeRatio}× mỗi từ. Tổng tham số MoE chỉ gấp <b>${moeTotalRatio}×</b> vì ~${nonFfnMoePct}% tham số (${fmtN(nonFfnParams)}) nằm ở Embedding + lớp Đầu ra, nên FFN chưa bị nghẽn cổ chai.<br/>
          <span style="color:#94a3b8">💡 Ghi chú: màu 4 chủ đề chỉ để quan sát — dữ liệu nhỏ này dùng từ vựng tách biệt theo chủ đề, khác với LLM lớn (panel X-Ray) nơi Router chia theo ngữ cảnh/cú pháp ngầm.</span>
        </div>`
      : `<div class="arena-summary-notes">
          • <code>Ep ${snap.epoch} · Step ${snap.step} · Seed #${msg.seed} · ${msg.totalSamples} tokens/epoch (100% shuffled)</code><br/>
          • <b>MoE Balance:</b> Aux = <code>${auxBadge}</code>, raw <code>ce·P = ${snap.moe.rawImbalance.toFixed(3)}</code> (1.00 = balanced, 2.00 = Top-2 collapse), <code>exp(H) = ${snap.moe.effectiveExperts.toFixed(2)}/4</code>, Spec = <b>${snap.moe.specializationPct}%</b>.<br/>
          <span style="color:#94a3b8">E + LM Head chiếm ${fmtN(nonFfnParams)} (~${nonFfnMoePct}% tổng tham số MoE), nên FFN không phải nút thắt chính trên corpus 64 câu: MoE (Aux BẬT) ngang ngửa Dense-18 và thua Dense-32; tắt Aux làm <code>exp(H)</code> tụt và CE Loss tăng. Màu miền chỉ phản ánh từ vựng tách biệt của bộ toy này (không trái ngược X-Ray ở LLM thật).</span>
        </div>`;

    summaryEl.innerHTML = `
      <table class="arena-cmp-table">
        <thead><tr>
          <th></th>
          <th>Dense nhỏ (18)</th>
          <th class="cur">MoE (18/32)</th>
          <th class="big">Dense to (32)</th>
        </tr></thead>
        <tbody>${tableRows}</tbody>
      </table>
      ${notesHtml}
    `;
  }
}
