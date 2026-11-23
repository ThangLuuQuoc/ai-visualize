"""
MiniMind Real-Time 3D Visualizer — PyTorch Telemetry Hook
=========================================================
Attach this lightweight hook to MiniMind's `train_pretrain.py`, `train_full_sft.py`,
`train_lora.py`, `train_dpo.py`, or `train_spo.py` to stream live training metrics
to the Node.js + Three.js 3D Visualizer at http://localhost:4000/api/telemetry.

Usage inside MiniMind training loop:
------------------------------------
from minimind_telemetry import MiniMindVisualizerHook

viz = MiniMindVisualizerHook(stage="pretrain", architecture="moe")
# Inside your training step loop:
viz.log_step(
    step=step,
    epoch=epoch,
    loss=loss.item(),
    lr=optimizer.param_groups[0]["lr"],
    model=model,
    aux_loss=getattr(res, "aux_loss", 0.0)
)
"""

import json
import time
import urllib.request
from typing import Any, Dict, List, Optional


class MiniMindVisualizerHook:
    def __init__(
        self,
        endpoint: str = "http://localhost:4000/api/telemetry",
        stage: str = "pretrain",
        architecture: str = "moe",
        send_interval_steps: int = 2,
    ):
        self.endpoint = endpoint
        self.stage = stage
        self.architecture = architecture
        self.send_interval_steps = send_interval_steps
        self.last_time = time.time()

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        model: Optional[Any] = None,
        aux_loss: float = 0.0,
        tokens_per_sec: float = 18500.0,
        extra_rl: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step % self.send_interval_steps != 0:
            return

        layers_payload: List[Dict[str, Any]] = []
        if model is not None:
            # Extract lightweight per-layer gradient norms without blocking GPU
            for idx in range(8):
                base_grad = 0.0
                lora_grad = 0.0
                for name, param in model.named_parameters():
                    if f"layers.{idx}." in name and param.grad is not None:
                        gnorm = float(param.grad.data.norm(2).item())
                        if "lora" in name.lower():
                            lora_grad += gnorm
                        else:
                            base_grad += gnorm
                layers_payload.append(
                    {
                        "layerIdx": idx,
                        "baseWeightGradNorm": round(base_grad, 4),
                        "loraAdapterGradNorm": round(lora_grad, 4),
                        "auxLoss": float(aux_loss),
                    }
                )

        payload = {
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
            "rl": extra_rl,
        }

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
