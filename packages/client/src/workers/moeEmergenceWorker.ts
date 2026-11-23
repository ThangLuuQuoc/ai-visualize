import {
  MoeEmergenceInMessage,
  RealEmergentMoEModel,
} from './moeEmergenceCore';

const moeModel = new RealEmergentMoEModel(42);
let running = false;
let timerId: ReturnType<typeof setInterval> | null = null;

function emit(): void {
  self.postMessage(moeModel.buildSnapshot(running));
}

function startLoop(): void {
  if (timerId !== null) clearInterval(timerId);
  timerId = setInterval(() => {
    if (!running) return;
    for (let i = 0; i < 3; i++) {
      moeModel.trainOneStep(64);
    }
    emit();
  }, 180);
}

self.onmessage = (ev: MessageEvent<MoeEmergenceInMessage>) => {
  const cmd = ev.data;
  switch (cmd.type) {
    case 'INIT':
      emit();
      break;
    case 'START':
      running = true;
      emit();
      startLoop();
      break;
    case 'PAUSE':
      running = false;
      emit();
      break;
    case 'STEP_BATCH': {
      const n = cmd.count ?? 15;
      for (let i = 0; i < n; i++) moeModel.trainOneStep(64);
      emit();
      break;
    }
    case 'RESEED': {
      const nextSeed = cmd.seed ?? Math.floor(10 + Math.random() * 990);
      moeModel.reseed(nextSeed);
      emit();
      break;
    }
    case 'SET_AUX_LOSS':
      moeModel.auxLossEnabled = cmd.enabled;
      emit();
      break;
    case 'SET_DOMAIN':
      moeModel.enabledDomains[cmd.domain] = cmd.enabled;
      emit();
      break;
    case 'SET_PROBE':
      moeModel.probeSentence = cmd.sentence.trim() || 'Hà Nội có Phở bò và Python tính Đạo hàm';
      emit();
      break;
  }
};
