import http from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import { ControlCommand, TelemetryFrame } from '@minimind-viz/shared';
import { MiniMindSimulator } from './simulator.js';

const PORT = Number(process.env.PORT || 4000);
const simulator = new MiniMindSimulator();
let lastLiveFrameTime = 0;

const server = http.createServer((req, res) => {
  // Enable CORS for browser client and local hooks
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

  // Receive live telemetry frames from MiniMind PyTorch hook
  if (req.method === 'POST' && req.url === '/api/telemetry') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      try {
        const incoming = JSON.parse(body) as Partial<TelemetryFrame>;
        const baseFrame = simulator.buildFrame();
        const mergedFrame: TelemetryFrame = {
          ...baseFrame,
          ...incoming,
          source: 'pytorch_live',
          layers:
            incoming.layers && incoming.layers.length > 0
              ? incoming.layers.map((l, i) => ({
                  ...baseFrame.layers[i],
                  ...l,
                }))
              : baseFrame.layers,
        };
        lastLiveFrameTime = Date.now();
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
  // Send immediate snapshot on connect
  ws.send(
    JSON.stringify({
      type: 'TELEMETRY_FRAME',
      frame: simulator.buildFrame(),
    })
  );

  ws.on('message', (raw) => {
    try {
      const cmd = JSON.parse(raw.toString()) as ControlCommand;
      simulator.applyCommand(cmd);
      broadcastFrame(simulator.buildFrame());
    } catch {
      // Ignore malformed messages
    }
  });
});

// Stream simulation frames every 350ms when no live PyTorch telemetry is active
setInterval(() => {
  const isLiveActive = Date.now() - lastLiveFrameTime < 3000;
  if (!isLiveActive) {
    const frame = simulator.tick();
    broadcastFrame(frame);
  }
}, 350);

server.listen(PORT, () => {
  console.log(`[MiniMind-3D-Server] WebSocket & Telemetry Server running on http://localhost:${PORT}`);
});
