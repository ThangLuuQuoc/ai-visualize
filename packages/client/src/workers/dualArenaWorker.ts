import { DualArenaEngine, DualArenaInMessage } from './dualArenaCore';

const engine = new DualArenaEngine(42);
let running = false;
let timerId: ReturnType<typeof setInterval> | null = null;

function emit(): void {
  self.postMessage(engine.getUpdateMessage(running));
}

function startLoop(): void {
  if (timerId !== null) clearInterval(timerId);
  timerId = setInterval(() => {
    if (!running) return;
    if (engine.epoch >= engine.MAX_EPOCHS) {
      running = false;
      emit();
      return;
    }
    engine.trainOneEpoch();
    if (engine.epoch >= engine.MAX_EPOCHS) {
      running = false;
    }
    emit();
  }, 550);
}

self.onmessage = (ev: MessageEvent<DualArenaInMessage>) => {
  const cmd = ev.data;
  switch (cmd.type) {
    case 'INIT':
      emit();
      break;
    case 'START':
      if (engine.epoch >= engine.MAX_EPOCHS) {
        engine.reseed(engine.seed);
      }
      running = true;
      emit();
      startLoop();
      break;
    case 'PAUSE':
      running = false;
      emit();
      break;
    case 'RUN_ONE_EPOCH':
      running = false;
      engine.trainOneEpoch();
      emit();
      break;
    case 'SELECT_EPOCH':
      running = false;
      self.postMessage(engine.getUpdateMessage(false, cmd.epoch));
      break;
    case 'RESEED': {
      running = false;
      const nextSeed = cmd.seed ?? Math.floor(10 + Math.random() * 990);
      engine.reseed(nextSeed);
      emit();
      break;
    }
    case 'SET_AUX_LOSS':
      engine.auxLossEnabled = cmd.enabled;
      emit();
      break;
  }
};
