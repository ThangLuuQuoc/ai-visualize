import { DualArenaEngine, DualArenaInMessage } from './dualArenaCore';

const engine = new DualArenaEngine(42);
let running = false;
let timerId: ReturnType<typeof setInterval> | null = null;

function emit(): void {
  self.postMessage(engine.getUpdateMessage(running));
}

function stopLoop(): void {
  running = false;
  if (timerId !== null) {
    clearInterval(timerId);
    timerId = null;
  }
}

function startLoop(): void {
  if (timerId !== null) clearInterval(timerId);
  timerId = setInterval(() => {
    if (!running || engine.epoch >= engine.MAX_EPOCHS) {
      stopLoop();
      emit();
      return;
    }
    engine.trainOneEpoch();
    if (engine.epoch >= engine.MAX_EPOCHS) {
      stopLoop();
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
      stopLoop();
      emit();
      break;
    case 'RUN_ONE_EPOCH':
      stopLoop();
      engine.trainOneEpoch();
      emit();
      break;
    case 'SELECT_EPOCH':
      stopLoop();
      self.postMessage(engine.getUpdateMessage(false, cmd.epoch));
      break;
    case 'RESEED': {
      stopLoop();
      const nextSeed = cmd.seed ?? Math.floor(10 + Math.random() * 990);
      engine.reseed(nextSeed);
      emit();
      break;
    }
    case 'SET_AUX_LOSS':
      stopLoop();
      engine.auxLossEnabled = cmd.enabled;
      engine.reseed(engine.seed);
      emit();
      break;
  }
};
