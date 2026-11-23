import http from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import {
  ControlCommand,
  MiniMindSimulator,
  TelemetryFrame,
  VramBreakdown,
} from '@minimind-viz/shared';

const PORT = Number(process.env.PORT || 4000);
const simulator = new MiniMindSimulator();
// How long to keep showing PyTorch live data after the last POST. The Python hook only sends every
// N steps (default 5), and a real MiniMind step can take >1s, so a short window (e.g. 3s) would make
// the UI flip between "PYTORCH LIVE" and simulated numbers. Override with LIVE_TIMEOUT_MS.
const LIVE_TIMEOUT_MS = Number(process.env.LIVE_TIMEOUT_MS || 30000);
let lastLiveFrameTime = 0;
let latestLivePayload: Partial<TelemetryFrame> | null = null;

function isLiveActive(): boolean {
  return latestLivePayload !== null && Date.now() - lastLiveFrameTime < LIVE_TIMEOUT_MS;
}

function mergeLiveTelemetry(
  baseFrame: TelemetryFrame,
  incoming: Partial<TelemetryFrame>
): TelemetryFrame {
  const totalParamsM = incoming.totalParamsM ?? baseFrame.totalParamsM;
  const trainableParamsM = incoming.trainableParamsM ?? baseFrame.trainableParamsM;
  const trainablePct = Number(((trainableParamsM / Math.max(0.1, totalParamsM)) * 100).toFixed(2));
  const stage = incoming.stage ?? baseFrame.stage;
  const isRl = stage === 'rl_dpo' || stage === 'rl_grpo';

  // Recompute VRAM breakdown using live parameter counts and live allocated VRAM
  const weightsMB = Math.round(totalParamsM * 2);
  const gradientsMB = Math.max(2, Math.round(trainableParamsM * 2));
  const optimizerMB = Math.max(8, Math.round(trainableParamsM * 12));
  const refModelMB = isRl ? weightsMB : 0;
  const reportedTotalMB =
    incoming.vramUsedMB ??
    weightsMB + gradientsMB + optimizerMB + refModelMB + baseFrame.vramBreakdown.activationsMB;
  const activationsMB = Math.max(
    64,
    reportedTotalMB - (weightsMB + gradientsMB + optimizerMB + refModelMB)
  );
  const totalMB = weightsMB + gradientsMB + optimizerMB + refModelMB + activationsMB;

  const vramBreakdown: VramBreakdown = {
    weightsMB,
    gradientsMB,
    optimizerMB,
    refModelMB,
    activationsMB,
    totalMB,
  };

  return {
    ...baseFrame,
    ...incoming,
    source: 'pytorch_live',
    step: incoming.step ?? baseFrame.step,
    epoch: incoming.epoch ?? baseFrame.epoch,
    loss: incoming.loss ?? baseFrame.loss,
    learningRate: incoming.learningRate ?? baseFrame.learningRate,
    tokensPerSec: incoming.tokensPerSec ?? baseFrame.tokensPerSec,
    totalParamsM,
    trainableParamsM,
    trainablePct,
    vramUsedMB: totalMB,
    vramBreakdown,
    // The hook sends `"rl": null` when it has no RL metrics; keep the simulated RL panel instead.
    rl: incoming.rl ?? baseFrame.rl,
    // Keep the continuous 3-phase animation state from simulator.tick()
    phase: baseFrame.phase,
    phaseProgress: baseFrame.phaseProgress,
    cycleSeconds: baseFrame.cycleSeconds,
    predictionDemo: {
      ...baseFrame.predictionDemo,
      batchAvgLoss: incoming.loss ?? baseFrame.predictionDemo.batchAvgLoss,
    },
    layers:
      incoming.layers && incoming.layers.length > 0
        ? incoming.layers.map((l, i) => ({
            ...baseFrame.layers[i],
            ...l,
          }))
        : baseFrame.layers,
  };
}

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'GET' && req.url === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify({
        status: 'ok',
        config: simulator.getConfig(),
        paused: simulator.isPaused(),
        speed: simulator.getSpeed(),
      })
    );
    return;
  }

  if (req.method === 'POST' && req.url === '/api/telemetry') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      try {
        const incoming = JSON.parse(body) as Partial<TelemetryFrame>;
        // Sync stage/arch/step WITHOUT resetting step to 1 or zeroing cycleSeconds
        simulator.syncFromLivePyTorch({
          stage: incoming.stage,
          architecture: incoming.architecture,
          step: incoming.step,
        });
        latestLivePayload = incoming;
        lastLiveFrameTime = Date.now();

        const mergedFrame = mergeLiveTelemetry(simulator.buildFrame(), incoming);
        broadcastFrame(mergedFrame);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end('Not Found');
});

const wss = new WebSocketServer({ server });

function broadcastFrame(frame: TelemetryFrame): void {
  const payload = JSON.stringify({ type: 'TELEMETRY_FRAME', frame });
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

wss.on('connection', (ws) => {
  const initialFrame =
    isLiveActive() && latestLivePayload
      ? mergeLiveTelemetry(simulator.buildFrame(), latestLivePayload)
      : simulator.buildFrame();

  ws.send(
    JSON.stringify({
      type: 'TELEMETRY_FRAME',
      frame: initialFrame,
    })
  );

  ws.on('message', (raw) => {
    try {
      const cmd = JSON.parse(raw.toString()) as ControlCommand;
      // If user clicks a stage button in the UI, exit stale live override
      if (cmd.type === 'SET_STAGE' || cmd.type === 'SET_ARCH' || cmd.type === 'RESET') {
        lastLiveFrameTime = 0;
        latestLivePayload = null;
      }
      simulator.applyCommand(cmd);
      broadcastFrame(simulator.buildFrame());
    } catch {
      // Ignore malformed messages
    }
  });
});

// Always advance the 3-phase animation clock every 120ms, even during live PyTorch streaming!
setInterval(() => {
  if (isLiveActive() && latestLivePayload) {
    const prevStep = latestLivePayload.step;
    const baseFrame = simulator.tick(0.12);
    // Keep the exact PyTorch step rather than auto-incrementing past it
    if (typeof prevStep === 'number') {
      simulator.syncFromLivePyTorch({ step: prevStep });
    }
    broadcastFrame(mergeLiveTelemetry(baseFrame, latestLivePayload));
  } else {
    broadcastFrame(simulator.tick(0.12));
  }
}, 120);

server.listen(PORT, () => {
  console.log(`[MiniMind-3D-Server] WebSocket & Telemetry Server running on http://localhost:${PORT}`);
});
