import {
  ArchitectureType,
  ControlCommand,
  TelemetryFrame,
  TrainingStage,
} from '@minimind-viz/shared';
import { LocalFallbackSimulator } from './localSimulator';
import { InspectableMetadata, MiniMindScene } from './scene/MiniMindScene';

const STAGE_INSIGHTS: Record<TrainingStage, { hint: string; html: string }> = {
  pretrain: {
    hint: 'Pretrain: Học tự giám sát trên toàn bộ 100% chuỗi token (không mask prompt).',
    html: `
      <b>1. Thu gọn Vocab (6,400) & FlashAttention:</b> Giảm >70% chi phí lớp Embedding/LM Head.<br/>
      <b>2. 100% Trọng số Mở khóa:</b> Sóng Gradient đỏ chảy ngược qua toàn bộ 8 Layers.<br/>
      <b>3. MoE Top-2 Routing:</b> Mỗi token chỉ kích hoạt 2/6 Routed Experts (~64M active / 198M total), giữ tốc độ train nhanh ngang mạng nhỏ.
    `,
  },
  sft: {
    hint: 'Full SFT: Áp dụng Prompt Loss Masking — chỉ tính Loss trên câu trả lời của Assistant.',
    html: `
      <b>1. Prompt Loss Masking:</b> Khi hạt token của <code>User</code> chạm cổng LM Head, tấm chắn đỏ bật lên chặn không sinh Gradient ngược.<br/>
      <b>2. Tập trung năng lực mô hình:</b> Mạng 64M không tốn dung lượng học thuộc câu hỏi mà hội tụ hoàn toàn vào chất lượng câu trả lời.
    `,
  },
  lora: {
    hint: 'LoRA (Low-Rank Adaptation): Đóng băng 100% mạng gốc W0, chỉ train nhánh phụ A × B (Rank-8).',
    html: `
      <b>1. Đóng băng Mạng Gốc (Frozen Glass):</b> Toàn bộ khối GQA & FFN/MoE chuyển sang băng trong suốt (0 gradient).<br/>
      <b>2. Nhánh đồng hồ cát A × B (r = 8):</b> Toàn bộ hạt Gradient màu tím chỉ chảy qua ống dẫn phụ bên phải (~1.24M tham số), giảm >55% VRAM và tăng tốc độ step lên ~29,500+ tok/s.
    `,
  },
  rl_dpo: {
    hint: 'RL (DPO - Direct Preference Optimization): So sánh trực tiếp cặp đáp án Thắng/Thua với Reference Model.',
    html: `
      <b>1. Không cần Reward Model & PPO Loop:</b> Tối ưu trực tiếp khoảng cách (Margin) giữa <code>y_win</code> (cầu xanh lá) và <code>y_lose</code> (cầu đỏ).<br/>
      <b>2. Neo KL-Divergence:</b> Làn Reference Model đóng băng bên trái giữ Policy Model không bị suy biến ngôn ngữ.
    `,
  },
  rl_grpo: {
    hint: 'RL (GRPO - Group Relative Policy Optimization): Lấy mẫu nhóm G=4 nhánh, loại bỏ hoàn toàn mô hình Critic.',
    html: `
      <b>1. Loại bỏ Critic Model (~50% VRAM):</b> Thay vì nuôi thêm 1 mạng Critic lớn như PPO, GRPO phát sinh chùm G = 4 câu trả lời (o₁ … o₄).<br/>
      <b>2. Chuẩn hóa Điểm Nhóm:</b> Tự tính Baseline từ trung bình nhóm Aᵢ = (rᵢ − μ) / σ, cực kỳ nhẹ cho GPU đơn.
    `,
  },
};

class MiniMindVisualizerApp {
  private scene: MiniMindScene;
  private localSim = new LocalFallbackSimulator();
  private ws: WebSocket | null = null;
  private wsConnected = false;
  private lossHistory: number[] = [];
  private selectedMeta: InspectableMetadata | null = null;

  constructor() {
    const canvasContainer = document.getElementById('canvas-container')!;
    const labelsContainer = document.getElementById('labels-container')!;
    this.scene = new MiniMindScene(canvasContainer, labelsContainer);

    this.scene.onNodeSelected((meta, frame) => {
      this.selectedMeta = meta;
      this.renderInspector(meta, frame);
    });

    this.bindUIEvents();
    this.connectWebSocket();

    // Local fallback loop when Node.js server is not running
    setInterval(() => {
      if (!this.wsConnected) {
        const frame = this.localSim.tick();
        this.onTelemetryFrame(frame, false);
      }
    }, 350);
  }

  private connectWebSocket(): void {
    try {
      const ws = new WebSocket('ws://localhost:4000');
      this.ws = ws;

      ws.onopen = () => {
        this.wsConnected = true;
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'TELEMETRY_FRAME' && msg.frame) {
            this.onTelemetryFrame(msg.frame as TelemetryFrame, true);
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        this.wsConnected = false;
        setTimeout(() => this.connectWebSocket(), 3500);
      };

      ws.onerror = () => {
        this.wsConnected = false;
      };
    } catch {
      this.wsConnected = false;
    }
  }

  private sendCommand(cmd: ControlCommand): void {
    this.localSim.applyCommand(cmd);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    } else {
      this.onTelemetryFrame(this.localSim.buildFrame(), false);
    }
  }

  private bindUIEvents(): void {
    // Architecture buttons
    const archBtns = document.querySelectorAll<HTMLButtonElement>('#arch-switcher button');
    archBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        archBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const arch = btn.dataset.arch as ArchitectureType;
        this.sendCommand({ type: 'SET_ARCH', architecture: arch });
      });
    });

    // Stage buttons
    const stageBtns = document.querySelectorAll<HTMLButtonElement>('#stage-switcher button');
    stageBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        stageBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const stage = btn.dataset.stage as TrainingStage;
        this.lossHistory = [];
        this.sendCommand({ type: 'SET_STAGE', stage });
      });
    });

    // Pause / Step / Camera Reset
    const btnPause = document.getElementById('btn-pause')!;
    btnPause.addEventListener('click', () => {
      this.sendCommand({ type: 'TOGGLE_PAUSE' });
      const paused = this.localSim.isPaused();
      btnPause.textContent = paused ? '▶ Resume' : '⏸ Pause';
    });

    document.getElementById('btn-step')!.addEventListener('click', () => {
      this.sendCommand({ type: 'STEP_ONCE' });
      btnPause.textContent = '▶ Resume';
    });

    document.getElementById('btn-cam-reset')!.addEventListener('click', () => {
      this.scene.resetCamera();
    });
  }

  private onTelemetryFrame(frame: TelemetryFrame, viaServer: boolean): void {
    this.scene.updateFromTelemetry(frame);

    // Connection badge
    const badge = document.getElementById('conn-badge')!;
    if (frame.source === 'pytorch_live') {
      badge.textContent = 'PYTORCH LIVE';
      badge.className = 'conn-badge live';
    } else if (viaServer) {
      badge.textContent = 'NODE.JS STREAM';
      badge.className = 'conn-badge sim';
    } else {
      badge.textContent = 'LOCAL SIM';
      badge.className = 'conn-badge sim';
    }

    // Phase pill
    const phasePill = document.getElementById('phase-pill')!;
    phasePill.textContent = frame.phase.toUpperCase().replace('_', ' ');
    phasePill.className = `phase-pill ${frame.phase}`;

    // Stage insights
    const info = STAGE_INSIGHTS[frame.stage];
    document.getElementById('token-stream-hint')!.textContent = info.hint;
    document.getElementById('stage-tag')!.textContent = frame.stage.toUpperCase();
    document.getElementById('optimization-insight')!.innerHTML = info.html;

    // Token chips
    const tokenListEl = document.getElementById('token-list')!;
    tokenListEl.innerHTML = frame.currentTokens
      .map((t, idx) => {
        const isActive = idx === frame.activeTokenIdx;
        const classes = ['token-chip'];
        if (isActive) classes.push('active');
        if (t.lossMasked) classes.push('masked');
        const routingInfo =
          frame.architecture === 'moe'
            ? `E${t.activeExperts.join(',E')}`
            : `RoPE ${t.ropeAngleRad}r`;
        return `
          <div class="${classes.join(' ')}">
            <span>${t.text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
            <span class="token-meta">${t.lossMasked ? 'MASKED (0 Loss)' : routingInfo}</span>
          </div>
        `;
      })
      .join('');

    // Right panel metrics
    document.getElementById('step-counter')!.textContent =
      `Step ${frame.step} • Epoch ${frame.epoch}`;
    document.getElementById('m-loss')!.textContent = frame.loss.toFixed(4);
    document.getElementById('m-lr')!.textContent = frame.learningRate.toExponential(2);
    document.getElementById('m-tps')!.textContent =
      `${frame.tokensPerSec.toLocaleString()} tok/s`;
    document.getElementById('m-vram')!.textContent =
      `${frame.vramUsedMB.toLocaleString()} MB`;
    document.getElementById('m-params')!.textContent =
      `${frame.trainableParamsM}M / ${frame.activeParamsM}M / ${frame.totalParamsM}M`;

    // Loss Chart
    this.lossHistory.push(frame.loss);
    if (this.lossHistory.length > 60) this.lossHistory.shift();
    this.drawLossChart();

    // Dynamic MoE Load or RL Rewards Panel
    this.renderDynamicPanel(frame);

    if (this.selectedMeta) {
      this.renderInspector(this.selectedMeta, frame);
    }
  }

  private drawLossChart(): void {
    const canvas = document.getElementById('loss-chart') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (this.lossHistory.length < 2) return;
    const minL = Math.min(...this.lossHistory) * 0.9;
    const maxL = Math.max(...this.lossHistory) * 1.05 + 0.05;

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    this.lossHistory.forEach((val, idx) => {
      const x = (idx / (this.lossHistory.length - 1)) * (w - 16) + 8;
      const y = h - 10 - ((val - minL) / (maxL - minL)) * (h - 20);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  private renderDynamicPanel(frame: TelemetryFrame): void {
    const titleEl = document.getElementById('dynamic-panel-title')!;
    const badgeEl = document.getElementById('dynamic-panel-badge')!;
    const bodyEl = document.getElementById('dynamic-panel-body')!;

    if (frame.stage === 'rl_dpo' && frame.rl) {
      titleEl.textContent = 'RL DPO — Chosen vs Rejected Margin';
      badgeEl.textContent = `KL: ${frame.rl.klDivergence}`;
      const winPct = Math.min(100, Math.max(10, (frame.rl.chosenReward + 1) * 35));
      const losePct = Math.min(100, Math.max(10, (frame.rl.rejectedReward + 2) * 25));
      bodyEl.innerHTML = `
        <div class="bar-row">
          <span class="bar-label">y_win (+)</span>
          <div class="bar-track"><div class="bar-fill" style="width:${winPct}%;background:#34d399;"></div></div>
          <span class="mono-text">${frame.rl.chosenReward}</span>
        </div>
        <div class="bar-row">
          <span class="bar-label">y_lose (-)</span>
          <div class="bar-track"><div class="bar-fill" style="width:${losePct}%;background:#fb7185;"></div></div>
          <span class="mono-text">${frame.rl.rejectedReward}</span>
        </div>
        <div class="bar-row">
          <span class="bar-label">Margin Δ</span>
          <span class="mono-text accent-emerald">+${frame.rl.rewardMargin} (No Critic Model needed)</span>
        </div>
      `;
      return;
    }

    if (frame.stage === 'rl_grpo' && frame.rl) {
      titleEl.textContent = 'RL GRPO — Group Relative Advantage (G=4)';
      badgeEl.textContent = `KL: ${frame.rl.klDivergence}`;
      bodyEl.innerHTML = frame.rl.groupAdvantages
        .map((adv, idx) => {
          const pct = Math.min(100, Math.max(12, (adv + 1.5) * 32));
          const color = adv >= 0 ? '#34d399' : '#fb7185';
          return `
            <div class="bar-row">
              <span class="bar-label">Branch o_${idx + 1}</span>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color};"></div></div>
              <span class="mono-text">${adv >= 0 ? '+' : ''}${adv}</span>
            </div>
          `;
        })
        .join('');
      return;
    }

    if (frame.architecture === 'moe') {
      const l0 = frame.layers[0];
      titleEl.textContent = 'MoE Expert Load (6 Routed + 1 Shared)';
      badgeEl.textContent = `Aux Loss: ${l0?.auxLoss ?? 0}`;
      const activeToken = frame.currentTokens[frame.activeTokenIdx];
      bodyEl.innerHTML = (l0?.expertLoads || [])
        .map((load, eIdx) => {
          const isTop = activeToken?.activeExperts.includes(eIdx);
          const pct = Math.min(100, Math.round(load * 320));
          const color = isTop ? '#f59e0b' : '#38bdf8';
          return `
            <div class="bar-row">
              <span class="bar-label">Expert E${eIdx} ${isTop ? '⚡' : ''}</span>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color};"></div></div>
              <span class="mono-text">${Math.round(load * 100)}%</span>
            </div>
          `;
        })
        .join('');
    } else {
      titleEl.textContent = 'GQA & Gradient Norm theo Layer';
      badgeEl.textContent = 'KV Savings: 75%';
      bodyEl.innerHTML = frame.layers
        .slice(0, 5)
        .map((l) => {
          const g = frame.stage === 'lora' ? l.loraAdapterGradNorm : l.baseWeightGradNorm;
          const pct = Math.min(100, Math.round(g * 130));
          const color = frame.stage === 'lora' ? '#e879f9' : '#22d3ee';
          return `
            <div class="bar-row">
              <span class="bar-label">Layer ${l.layerIdx}</span>
              <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color};"></div></div>
              <span class="mono-text">||g||=${g}</span>
            </div>
          `;
        })
        .join('');
    }
  }

  private renderInspector(meta: InspectableMetadata, frame: TelemetryFrame | null): void {
    document.getElementById('inspect-badge')!.textContent = `LAYER ${meta.layerIdx}`;
    const layer = frame?.layers[meta.layerIdx];
    const baseGrad = layer?.baseWeightGradNorm ?? 0;
    const loraGrad = layer?.loraAdapterGradNorm ?? 0;

    document.getElementById('inspector-body')!.innerHTML = `
      <div style="font-weight:700;color:#38bdf8;margin-bottom:4px;">${meta.title}</div>
      <div class="mono-text" style="color:#fbbf24;margin-bottom:6px;">Shape: ${meta.shape}</div>
      <p style="margin-bottom:8px;">${meta.description}</p>
      <div class="mono-text" style="font-size:11px;color:#94a3b8;">
        • Base Weight Grad Norm: <b>${baseGrad}</b> ${baseGrad === 0 ? '(FROZEN ❄️)' : ''}<br/>
        • LoRA Adapter Grad Norm: <b>${loraGrad}</b>
      </div>
    `;
  }
}

new MiniMindVisualizerApp();
