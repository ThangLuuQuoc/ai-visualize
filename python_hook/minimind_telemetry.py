"""
MiniMind Real-Time 3D Visualizer — PyTorch Telemetry Hook
=========================================================
Attach this lightweight hook to MiniMind's `train_pretrain.py`, `train_full_sft.py`,
`train_lora.py`, `train_dpo.py`, or `train_spo.py` to stream live training metrics
to the Node.js + Three.js 3D Visualizer at http://localhost:4000/api/telemetry.

Key performance guarantees:
- Single pass over `model.named_parameters()` per logged step.
- Accumulates squared L2 norms on GPU (`||g||_2^2`) and transfers a small
  `(num_layers, 2)` tensor to CPU with a single `.cpu().tolist()` call,
  avoiding per-parameter `.item()` GPU synchronization stalls.
- Computes the exact L2 layer norm: sqrt(sum_i ||g_i||_2^2).
"""

import json
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


class MiniMindVisualizerHook:
    def __init__(
        self,
        endpoint: str = "http://localhost:4000/api/telemetry",
        stage: str = "pretrain",
        architecture: str = "moe",
        num_layers: int = 8,
        send_interval_steps: int = 5,
    ):
        self.endpoint = endpoint
        self.stage = stage
        self.architecture = architecture
        self.num_layers = num_layers
        self.send_interval_steps = send_interval_steps

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        model: Optional[Any] = None,
        aux_loss: float = 0.0,
        tokens_per_sec: float = 18400.0,
        extra_rl: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step % self.send_interval_steps != 0:
            return

        layers_payload: List[Dict[str, Any]] = []
        vram_used_mb: Optional[int] = None
        total_params_m: Optional[float] = None
        trainable_params_m: Optional[float] = None

        if model is not None and torch is not None:
            with torch.no_grad():
                first_param = next(model.parameters(), None)
                device = first_param.device if first_param is not None else "cpu"
                # Column 0: base weights sum of squared norms; Column 1: LoRA adapter sum of squared norms
                sq_norms = torch.zeros((self.num_layers, 2), dtype=torch.float32, device=device)

                total_p = 0
                trainable_p = 0
                for name, param in model.named_parameters():
                    n_el = param.numel()
                    total_p += n_el
                    if param.requires_grad:
                        trainable_p += n_el
                    if param.grad is None:
                        continue
                    match = _LAYER_RE.search(name)
                    if not match:
                        continue
                    layer_idx = int(match.group(1))
                    if 0 <= layer_idx < self.num_layers:
                        col = 1 if "lora" in name.lower() else 0
                        g_norm = torch.linalg.vector_norm(param.grad.detach(), 2)
                        sq_norms[layer_idx, col] += g_norm * g_norm

                # Single GPU -> CPU synchronization for all layers
                l2_norms = torch.sqrt(sq_norms).cpu().tolist()
                total_params_m = round(total_p / 1e6, 2)
                trainable_params_m = round(trainable_p / 1e6, 2)

                if torch.cuda.is_available():
                    vram_used_mb = int(torch.cuda.memory_allocated() // (1024 * 1024))

                for idx in range(self.num_layers):
                    layers_payload.append(
                        {
                            "layerIdx": idx,
                            "baseWeightGradNorm": round(float(l2_norms[idx][0]), 4),
                            "loraAdapterGradNorm": round(float(l2_norms[idx][1]), 4),
                            "auxLoss": float(aux_loss),
                        }
                    )

        payload: Dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
            "step": step,
            "epoch": epoch,
            "stage": self.stage,
            "architecture": self.architecture,
            "source": "pytorch_live",
            "loss": float(loss),
            "learningRate": float(lr),
            "tokensPerSec": float(tokens_per_sec),
            "layers": layers_payload,
        }
        if extra_rl is not None:
            payload["rl"] = extra_rl
        if vram_used_mb is not None:
            payload["vramUsedMB"] = vram_used_mb
        if total_params_m is not None:
            payload["totalParamsM"] = total_params_m
        if trainable_params_m is not None:
            payload["trainableParamsM"] = trainable_params_m

        try:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.15)
        except Exception:
            # Never interrupt training if visualizer server is offline
            pass
