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

  constructor() {
    this.buildDenseSvg();
    this.buildMoeSvg();
    this.ensureStaticContainersBuilt();
    this.bindEvents();
  }

  public open(): void {
    this.isOpenState = true;
    this.modal.classList.remove('hidden');
    this.ensureWorker();
    if (this.latestMsg) this.render(this.latestMsg);
  }

  public close(): void {
    if (!this.isOpenState) return;
    this.isOpenState = false;
    this.modal.classList.add('hidden');
    this.worker?.postMessage({ type: 'PAUSE' });
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

    // Shared top pipeline: x -> RMSNorm -> Attention -> + -> FFN(18/18) -> y
    svgEl('path', { d: 'M 42 95 L 86 95', stroke: '#94a3b8', 'stroke-width': 2.5, fill: 'none' }, svg);
    svgEl('path', { d: 'M 186 95 L 222 95', stroke: '#38bdf8', 'stroke-width': 3, fill: 'none' }, svg);
    svgEl('path', { d: 'M 366 95 L 414 95', stroke: '#818cf8', 'stroke-width': 3, fill: 'none' }, svg);
    // Residual skip arrow over FFN
    svgEl(
      'path',
      {
        d: 'M 196 95 L 196 28 L 392 28 L 392 95',
        stroke: '#64748b',
        'stroke-width': 1.5,
        'stroke-dasharray': '4 3',
        fill: 'none',
      },
      svg
    );

    svgEl('circle', { cx: 30, cy: 95, r: 15, fill: '#0e7490', stroke: '#22d3ee', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 30, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, 'font-weight': 800 }, svg).textContent = 'x';
    this.denseTokText = svgEl('text', { x: 48, y: 66, 'text-anchor': 'middle', fill: '#67e8f9', 'font-size': 11, 'font-weight': 800 }, svg);

    // Identical Attention block
    svgEl('rect', { x: 86, y: 66, width: 100, height: 58, rx: 7, fill: 'rgba(59,130,246,0.22)', stroke: '#3b82f6', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 136, y: 90, 'text-anchor': 'middle', fill: '#dbeafe', 'font-size': 10.5, 'font-weight': 800 }, svg).textContent = 'Attention';
    svgEl('text', { x: 136, y: 105, 'text-anchor': 'middle', fill: '#93c5fd', 'font-size': 9 }, svg).textContent = '(Giống hệt MoE)';

    // Single monolithic SwiGLU FFN block (18/18 active)
    this.denseFfnRect = svgEl('rect', { x: 222, y: 50, width: 144, height: 90, rx: 9, fill: 'rgba(99,102,241,0.34)', stroke: '#818cf8', 'stroke-width': 2.5 }, svg);
    svgEl('text', { x: 294, y: 84, 'text-anchor': 'middle', fill: '#fff', 'font-size': 12, 'font-weight': 800 }, svg).textContent = '1 Khối FFN Duy Nhất';
    this.denseFfnLabel = svgEl('text', { x: 294, y: 102, 'text-anchor': 'middle', fill: '#c7d2fe', 'font-size': 10 }, svg);
    svgEl('text', { x: 294, y: 120, 'text-anchor': 'middle', fill: '#a5b4fc', 'font-size': 9.5 }, svg).textContent = '100% nơ-ron luôn sáng';

    svgEl('circle', { cx: 428, cy: 95, r: 14, fill: '#334155', stroke: '#e2e8f0', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 428, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, 'font-weight': 800 }, svg).textContent = 'y';
  }

  private buildMoeSvg(): void {
    const svg = this.moeSvg;
    svg.innerHTML = '';

    const expY = [24, 62, 100, 138];
    const sharedY = 172;

    svgEl('path', { d: 'M 36 95 L 68 95', stroke: '#94a3b8', 'stroke-width': 2, fill: 'none' }, svg);
    // Attention -> Router
    svgEl('path', { d: 'M 148 95 L 176 80', stroke: '#fbbf24', 'stroke-width': 2.2, fill: 'none' }, svg);
    // Attention -> Shared (no router)
    svgEl('path', { d: `M 148 112 C 165 ${sharedY}, 210 ${sharedY}, 268 ${sharedY}`, stroke: SHARED_COLOR, 'stroke-width': 2.2, fill: 'none' }, svg);
    svgEl('path', { d: `M 340 ${sharedY} C 380 ${sharedY}, 390 95, 404 95`, stroke: SHARED_COLOR, 'stroke-width': 2.2, fill: 'none' }, svg);
    svgEl('path', { d: 'M 426 95 L 438 95', stroke: '#e2e8f0', 'stroke-width': 2.5, fill: 'none' }, svg);

    for (let e = 0; e < 4; e++) {
      this.moeInPaths.push(
        svgEl('path', { d: `M 218 80 C 242 80, 240 ${expY[e]}, 268 ${expY[e]}`, fill: 'none' }, svg)
      );
      this.moeOutPaths.push(
        svgEl('path', { d: `M 340 ${expY[e]} C 378 ${expY[e]}, 386 95, 404 95`, fill: 'none' }, svg)
      );
    }

    svgEl('circle', { cx: 24, cy: 95, r: 13, fill: '#0e7490', stroke: '#22d3ee', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 24, y: 99, 'text-anchor': 'middle', fill: '#fff', 'font-size': 10.5, 'font-weight': 800 }, svg).textContent = 'x';
    this.moeTokText = svgEl('text', { x: 42, y: 66, 'text-anchor': 'middle', fill: '#67e8f9', 'font-size': 10.5, 'font-weight': 800 }, svg);

    // Identical Attention block
    svgEl('rect', { x: 68, y: 68, width: 80, height: 54, rx: 6, fill: 'rgba(59,130,246,0.22)', stroke: '#3b82f6', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 108, y: 92, 'text-anchor': 'middle', fill: '#dbeafe', 'font-size': 10, 'font-weight': 800 }, svg).textContent = 'Attention';
    svgEl('text', { x: 108, y: 106, 'text-anchor': 'middle', fill: '#93c5fd', 'font-size': 8.5 }, svg).textContent = '(Giống Dense)';

    // Router
    svgEl('circle', { cx: 196, cy: 80, r: 20, fill: '#b45309', stroke: '#fbbf24', 'stroke-width': 2 }, svg);
    svgEl('text', { x: 196, y: 83, 'text-anchor': 'middle', fill: '#fff', 'font-size': 9.5, 'font-weight': 800 }, svg).textContent = 'Router';

    // 4 Routed Experts
    for (let e = 0; e < 4; e++) {
      this.moeExpRects.push(
        svgEl('rect', { x: 268, y: expY[e] - 13, width: 72, height: 26, rx: 6, 'stroke-width': 1.8 }, svg)
      );
      this.moeExpStatus.push(
        svgEl('text', { x: 304, y: expY[e] + 4, 'text-anchor': 'middle', 'font-size': 9.8, 'font-weight': 800 }, svg)
      );
    }

    // Shared Expert
    svgEl('rect', { x: 268, y: sharedY - 12, width: 72, height: 24, rx: 6, fill: 'rgba(16,185,129,0.25)', stroke: SHARED_COLOR, 'stroke-width': 1.8 }, svg);
    svgEl('text', { x: 304, y: sharedY + 4, 'text-anchor': 'middle', fill: '#d1fae5', 'font-size': 9.5, 'font-weight': 800 }, svg).textContent = 'Shared (4)';

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
      `18/18 active (${snap.dense.ffnParams} FFN params) • Loss: ${snap.dense.ceLoss.toFixed(3)}`;
    this.denseTokText.textContent = `“${probeRef.word}”`;
    this.denseFfnLabel.textContent = `Tính toàn bộ 18/18 nơ-ron cho “${probeRef.word}”`;
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
      `18 active / 32 total (${snap.moe.ffnParams} FFN params) • Loss: ${snap.moe.ceLoss.toFixed(3)}`;
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
    const padL = 36;
    const padR = 36;
    const padT = 10;
    const padB = 16;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '10px Inter, Segoe UI, sans-serif';

    ctx.strokeStyle = 'rgba(148,163,184,0.16)';
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    const hist = msg.history;
    if (hist.length < 2) {
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('Bấm ▶ Chạy liên tục hoặc ⏭ Chạy 1 Epoch rồi dừng để xem đường học của Dense vs MoE...', padL + 8, h / 2 + 4);
      return;
    }

    const maxX = Math.max(1, hist.length - 1);
    const xOf = (i: number) => padL + (i / maxX) * (w - padL - padR);
    const maxLoss = Math.max(2.5, hist[0].denseCeLoss, hist[0].moeCeLoss);
    const yLoss = (l: number) => padT + (1 - Math.min(maxLoss, l) / maxLoss) * (h - padT - padB);
    const ySpec = (pct: number) => padT + (1 - Math.min(100, pct) / 100) * (h - padT - padB);

    // Scrubber cursor line
    const cursorX = xOf(msg.viewedEpoch);
    ctx.strokeStyle = 'rgba(254, 240, 138, 0.55)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cursorX, padT);
    ctx.lineTo(cursorX, h - padB);
    ctx.stroke();
    ctx.setLineDash([]);

    const drawSeries = (color: string, fn: (pt: (typeof hist)[0]) => number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.1;
      ctx.beginPath();
      hist.forEach((pt, i) => {
        const y = fn(pt);
        if (i === 0) ctx.moveTo(xOf(i), y);
        else ctx.lineTo(xOf(i), y);
      });
      ctx.stroke();
    };

    drawSeries('#818cf8', (pt) => yLoss(pt.denseCeLoss));
    drawSeries('#f59e0b', (pt) => yLoss(pt.moeCeLoss));
    drawSeries('#34d399', (pt) => ySpec(pt.moeSpecPct));

    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Ep 0', padL, h - 3);
    ctx.fillText(`Ep ${hist[hist.length - 1].epoch}`, w - padR - 30, h - 3);
    ctx.fillStyle = '#34d399';
    ctx.fillText(`${msg.current.moe.specializationPct}%`, w - padR + 4, padT + 10);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText(`${msg.current.moe.ceLoss.toFixed(2)}`, 2, padT + 10);
  }

  private renderSummary(msg: DualArenaUpdateMessage): void {
    const snap = msg.current;
    const summaryEl = document.getElementById('arena-summary')!;
    if (this.isSimple) {
      summaryEl.innerHTML =
        `<b>⚖️ So sánh công bằng ở Epoch ${snap.epoch} (cùng 18 nơ-ron tính mỗi từ):</b><br/>` +
        `• <b>Bên trái (Dense):</b> Chỉ có 1 phòng ban 18 người nên từ nào cũng đi chung 1 đường.<br/>` +
        `• <b>Bên phải (MoE):</b> Chia thành 4 chuyên khoa + 1 đa khoa (tổng 32 người, bộ nhớ ~1.8×) → mỗi từ vẫn chỉ gọi 18 người, nhưng độ phân hóa đạt <b>${snap.moe.specializationPct}%</b> (${snap.moe.effectiveExperts}/4 chuyên gia hoạt động).<br/>` +
        `<span style="color:#94a3b8">💡 Trên dữ liệu nhỏ 64 câu, Dense đủ chỗ ghi nhớ nhanh; lợi thế MoE bùng nổ khi dữ liệu lớn gấp nghìn lần khiến 1 khối Dense bị quá tải.</span>`;
    } else {
      summaryEl.innerHTML =
        `<b>🔬 Iso-FLOPs Telemetry (Epoch ${snap.epoch}, Step ${snap.step}, Seed #${msg.seed}):</b><br/>` +
        `• <b>Dense FFN:</b> 18 active / 18 total (<code>${snap.dense.ffnParams}</code> params) → CE Loss = <b>${snap.dense.ceLoss.toFixed(4)}</b>.<br/>` +
        `• <b>MoE FFN:</b> 18 active / 32 total (<code>${snap.moe.ffnParams}</code> params, ~1.97×) → CE Loss = <b>${snap.moe.ceLoss.toFixed(4)}</b>, Aux = <code>${snap.moe.auxLoss}</code>, Eff = <b>${snap.moe.effectiveExperts}/4</b>, Spec = <b>${snap.moe.specializationPct}%</b>.<br/>` +
        `<span style="color:#94a3b8">Tua thanh trượt Epoch để quan sát pha phá vỡ đối xứng (Ep 4–10) và thử tắt Aux Loss từ Ep 0 để thấy Expert Collapse ngay trong cùng đấu trường.</span>`;
    }
  }
}
