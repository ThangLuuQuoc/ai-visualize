"""
End-to-End Live Telemetry Test Script for MiniMind 3D Visualizer
================================================================
Simulates a slow PyTorch training run (sending POST /api/telemetry every 2-3 seconds)
to verify:
  1. `PYTORCH LIVE` badge stays rock-solid (no flickering to NODE.JS STREAM during 2-3s intervals).
  2. Real step counter advances cleanly (never resets to Step 1).
  3. Real loss curve, per-layer L2 gradient norms, and VRAM breakdown update accurately.

Usage:
  python python_hook/demo_pytorch_stream.py --stage sft --arch moe --interval 2.5 --steps 15
  python python_hook/demo_pytorch_stream.py --stage lora --arch dense --interval 2.0 --steps 12
"""

import argparse
import json
import math
import time
import urllib.request
from typing import Any, Dict, List


def send_telemetry(endpoint: str, payload: Dict[str, Any]) -> bool:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        return resp.status == 200


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream simulated PyTorch telemetry to MiniMind 3D Visualizer.")
    parser.add_argument("--endpoint", default="http://localhost:4000/api/telemetry")
    parser.add_argument("--stage", default="sft", choices=["pretrain", "sft", "lora", "rl_dpo", "rl_grpo"])
    parser.add_argument("--arch", default="moe", choices=["dense", "moe"])
    parser.add_argument("--interval", type=float, default=2.5, help="Seconds between telemetry POSTs")
    parser.add_argument("--steps", type=int, default=15, help="Total training steps to stream")
    args = parser.parse_args()

    is_moe = args.arch == "moe"
    is_lora = args.stage == "lora"
    total_params_m = 145.0 if is_moe else 25.8
    trainable_params_m = (0.16 if is_moe else 0.13) if is_lora else total_params_m

    # Realistic VRAM footprint matching MiniMind2 bfloat16 + FP32 AdamW
    weights_mb = int(round(total_params_m * 2))
    grads_mb = max(2, int(round(trainable_params_m * 2)))
    opt_mb = max(8, int(round(trainable_params_m * 12)))
    ref_mb = weights_mb if args.stage in ("rl_dpo", "rl_grpo") else 0
    act_mb = 520 if is_moe else 310
    vram_used_mb = weights_mb + grads_mb + opt_mb + ref_mb + act_mb

    start_loss = {"pretrain": 6.60, "sft": 2.35, "lora": 1.55, "rl_dpo": 0.69, "rl_grpo": 0.68}[args.stage]
    end_loss = {"pretrain": 2.30, "sft": 1.05, "lora": 0.85, "rl_dpo": 0.30, "rl_grpo": 0.28}[args.stage]
    base_lr = {"pretrain": 5e-4, "sft": 5e-5, "lora": 1e-4, "rl_dpo": 5e-6, "rl_grpo": 2e-6}[args.stage]

    print(
        f"[Demo PyTorch Stream] Streaming {args.steps} steps ({args.stage.upper()} / {args.arch.upper()}) "
        f"every {args.interval}s -> {args.endpoint}"
    )

    for step in range(1, args.steps + 1):
        progress = (step - 1) / max(1, args.steps - 1)
        lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        loss = round(end_loss + (start_loss - end_loss) * math.exp(-progress * 2.6) + math.sin(step) * 0.02, 4)

        layers: List[Dict[str, Any]] = []
        for idx in range(8):
            depth_scale = 1.0 - idx / 12.0
            base_grad = 0.0 if is_lora else round(0.38 * depth_scale * (1.0 - 0.3 * progress), 4)
            lora_grad = round(0.58 * depth_scale * (1.0 - 0.25 * progress), 4) if is_lora else 0.0
            layers.append(
                {
                    "layerIdx": idx,
                    "baseWeightGradNorm": base_grad,
                    "loraAdapterGradNorm": lora_grad,
                    "auxLoss": round(0.018 * math.exp(-progress * 1.8) + 0.002, 4) if is_moe else 0.0,
                }
            )

        payload: Dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
            "step": step * 5,  # Simulates send_interval_steps=5
            "epoch": 1 + (step * 5) // 40,
            "stage": args.stage,
            "architecture": args.arch,
            "source": "pytorch_live",
            "loss": loss,
            "learningRate": float(f"{lr:.3e}"),
            "tokensPerSec": 19200 if is_moe else 27400,
            "vramUsedMB": vram_used_mb,
            "totalParamsM": total_params_m,
            "trainableParamsM": trainable_params_m,
            "layers": layers,
        }

        try:
            ok = send_telemetry(args.endpoint, payload)
            print(
                f"  -> Sent Step {payload['step']:>3d} | Loss={loss:.4f} | LR={payload['learningRate']:.2e} "
                f"| VRAM={vram_used_mb}MB | ok={ok}"
            )
        except Exception as exc:
            print(f"  !! Failed to POST telemetry: {exc}")

        if step < args.steps:
            time.sleep(args.interval)

    print("[Demo PyTorch Stream] Completed.")


if __name__ == "__main__":
    main()
