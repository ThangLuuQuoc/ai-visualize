import { MiniMindSimulator, TelemetryFrame } from '@minimind-viz/shared';

const STAGE_NAME: Record<TelemetryFrame['stage'], string> = {
  pretrain: 'Pretrain',
  sft: 'Full SFT',
  lora: 'LoRA',
  rl_dpo: 'DPO',
  rl_grpo: 'GRPO',
};

function fmtM(v: number): string {
  return `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)}M`;
}

function fmtMB(v: number): string {
  // Same unit as the VRAM card above (MB, unit shown in the row label), so numbers match by eye.
  return v.toLocaleString('en-US');
}

/** Cache key: the card only changes when one of these changes. */
export function archCompareKey(frame: TelemetryFrame, isSimple: boolean): string {
  const eff = frame.moeBalance ? Math.round(frame.moeBalance.effectiveExperts * 10) : 0;
  return `${frame.stage}|${frame.architecture}|${isSimple}|${eff}`;
}

/**
 * Dense vs MoE, done fairly. Comparing Dense-Small (25.8M) with MoE (145M, 45.2M active) alone would
 * suggest MoE costs MORE compute per token (45.2M > 25.8M) because the two models have different
 * widths (d=512 vs d=640). The point of MoE only shows against a Dense model of the SAME total size,
 * so a third, hypothetical "Dense 145M" column is added.
 */
export function renderArchComparison(el: HTMLElement, frame: TelemetryFrame, isSimple: boolean): void {
  const dense = MiniMindSimulator.computeFootprint('dense', frame.stage);
  const moe = MiniMindSimulator.computeFootprint('moe', frame.stage);
  // Hypothetical dense model with MoE's total size: every token uses all 145M params.
  // Weights/grads/AdamW scale with total params (same as MoE); activations scale with what each
  // token actually computes, so they are at least MoE's. We only claim a lower bound.
  const bigDenseTotal = moe.totalParamsM;
  const bigDenseVramMin = moe.vramBreakdown.totalMB;

  const curDense = frame.architecture === 'dense' ? 'cur' : '';
  const curMoe = frame.architecture === 'moe' ? 'cur' : '';
  const ratio = (active: number) => `${Number((active / dense.activeParamsM).toFixed(2))}×`;

  const rows = isSimple
    ? [
        ['🧠 Tổng “núm vặn” (sức chứa kiến thức)', fmtM(dense.totalParamsM), fmtM(moe.totalParamsM), fmtM(bigDenseTotal), ['', 'good', '']],
        ['⚡ Núm vặn phải tính cho MỖI từ', fmtM(dense.activeParamsM), fmtM(moe.activeParamsM), fmtM(bigDenseTotal), ['', 'good', 'bad']],
        ['⏱ Công sức mỗi từ (so Dense nhỏ)', '1×', ratio(moe.activeParamsM), ratio(bigDenseTotal), ['', 'good', 'bad']],
        [`💾 Bộ nhớ GPU khi train, MB (${STAGE_NAME[frame.stage]})`, fmtMB(dense.vramBreakdown.totalMB), fmtMB(moe.vramBreakdown.totalMB), `≥ ${fmtMB(bigDenseVramMin)}`, ['', 'bad', 'bad']],
      ]
    : [
        ['Tổng tham số', fmtM(dense.totalParamsM), fmtM(moe.totalParamsM), fmtM(bigDenseTotal), ['', 'good', '']],
        ['Tham số kích hoạt / token', fmtM(dense.activeParamsM), fmtM(moe.activeParamsM), fmtM(bigDenseTotal), ['', 'good', 'bad']],
        ['FLOPs / token (tương đối)', '1×', ratio(moe.activeParamsM), ratio(bigDenseTotal), ['', 'good', 'bad']],
        ['hidden_size', '512', '640', '—', ['', '', '']],
        [`VRAM train, MB (${STAGE_NAME[frame.stage]}, bf16 + AdamW)`, fmtMB(dense.vramBreakdown.totalMB), fmtMB(moe.vramBreakdown.totalMB), `≥ ${fmtMB(bigDenseVramMin)}`, ['', 'bad', 'bad']],
      ];

  const bodyRows = rows
    .map((r) => {
      const cls = r[4] as string[];
      return `<tr>
        <td>${r[0]}</td>
        <td class="${curDense} ${cls[0]}">${r[1]}</td>
        <td class="${curMoe} ${cls[1]}">${r[2]}</td>
        <td class="hypo ${cls[2]}">${r[3]}</td>
      </tr>`;
    })
    .join('');

  const takeaway = isSimple
    ? `<b>MoE = kiến thức của mạng ${fmtM(moe.totalParamsM)}, nhưng mỗi từ chỉ tốn sức như mạng ${fmtM(moe.activeParamsM)}.</b> Cái giá: GPU vẫn phải <b>chứa đủ cả ${fmtM(moe.totalParamsM)}</b> (cả 4 chuyên gia), nên bộ nhớ gần bằng mạng Dense to.`
    : `<b>Conditional computation:</b> tách “dung lượng” (tổng tham số) khỏi “chi phí tính” (tham số kích hoạt). So với Dense cùng 145M, MoE giảm FLOPs/token ~${(bigDenseTotal / moe.activeParamsM).toFixed(1)}×, nhưng weights + grad + AdamW vẫn tỉ lệ với <i>tổng</i> tham số.`;

  const bal = frame.moeBalance;
  const warn =
    bal && frame.architecture === 'moe' && bal.effectiveExperts < 3.2
      ? `<div class="archcmp-warn">⚠ ${
          isSimple
            ? `${bal.auxLossEnabled ? 'Đang hồi phục sau khi tắt Aux Loss' : 'Đang tắt Aux Loss'}: thực chất chỉ ~<b>${bal.effectiveExperts.toFixed(1)}/4</b> chuyên gia được dùng → vẫn trả bộ nhớ cho 4 nhưng sức chứa thật ít hơn. Đó là lý do cần Aux Loss.`
            : `Load collapse: số expert hiệu dụng exp(H(load)) ≈ ${bal.effectiveExperts.toFixed(2)}/4 → dung lượng thực giảm, VRAM giữ nguyên.`
        }</div>`
      : '';

  const foot = isSimple
    ? '<div class="archcmp-foot">Cột 3 là mạng giả định để so cho công bằng: Dense nhỏ (25.8M) và MoE (145M) có kích thước khác nhau, nên so trực tiếp sẽ thấy MoE tốn sức hơn Dense nhỏ.</div>'
    : '<div class="archcmp-foot">Cột 3 là mốc giả định cùng tổng tham số với MoE. VRAM ghi “≥” vì activation của mạng Dense đó còn lớn hơn MoE. Số MoE theo MiniMind2-MoE (4 routed + 1 shared, Top-2).</div>';

  el.innerHTML = `
    <table class="archcmp-table">
      <thead><tr>
        <th></th>
        <th class="${curDense}">Dense nhỏ${curDense ? ' 👈' : ''}</th>
        <th class="${curMoe}">MoE${curMoe ? ' 👈' : ''}</th>
        <th class="hypo">Dense to<br/>(giả định)</th>
      </tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>
    <div class="archcmp-takeaway">${takeaway}</div>
    ${warn}
    ${foot}
  `;
}
