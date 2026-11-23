import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MiniMindSimulator, TelemetryFrame, TrainingStage } from '@minimind-viz/shared';

export interface InspectableMetadata {
  title: string;
  layerIdx: number;
  kind: 'embedding' | 'gqa' | 'dense_ffn' | 'moe_router' | 'moe_expert' | 'lora_adapter' | 'rl_node';
  shape: string;
  simpleDesc: string;
  technicalDesc: string;
}

interface Layer3DNodes {
  group: THREE.Group;
  zPos: number;
  gqaMesh: THREE.Mesh;
  kvCacheMesh: THREE.Mesh;
  denseFfnMesh: THREE.Mesh;
  moeGroup: THREE.Group;
  routerMesh: THREE.Mesh;
  sharedExpertMesh: THREE.Mesh;
  routedExpertMeshes: THREE.Mesh[];
  expertLoadBars: THREE.Mesh[];
  routerBeams: THREE.Line[];
  loraGroup: THREE.Group;
  loraDownMesh: THREE.Mesh;
  loraUpMesh: THREE.Mesh;
  refGhostMesh: THREE.Mesh;
  klBeam: THREE.Line;
}

interface FloatingLabel {
  el: HTMLDivElement;
  worldPos: THREE.Vector3;
  isLockBadge?: boolean;
  isRopeLabel?: boolean;
}

export class MiniMindScene {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private clock = new THREE.Clock();
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();

  private layerNodes: Layer3DNodes[] = [];
  private ropeGroup!: THREE.Group;
  private lossMaskGate!: THREE.Mesh;
  private lossBurstRing!: THREE.Mesh;
  private dpoArenaGroup!: THREE.Group;
  private grpoArenaGroup!: THREE.Group;
  private grpoBranches: THREE.Mesh[] = [];

  // Synchronized 60fps wavefront objects
  private tokenPacketMeshes: THREE.Mesh[] = [];
  private backwardWaveBar!: THREE.Mesh;
  private baseActivationGradWaveBar!: THREE.Mesh;

  private labels: FloatingLabel[] = [];
  private labelsContainer: HTMLElement;
  private latestFrame: TelemetryFrame | null = null;
  private lastStage: TrainingStage | null = null;
  private suppressAutoFocusForStage: TrainingStage | null = null;

  // Continuous 60fps client-side cycle clock synced to server telemetry
  private localCycleSeconds = 0;
  private stepDurationSec = 4.8;
  private currentSpeed = 1.0;
  private isPaused = false;

  // Exact 0 -> 1 camera fly-to state using lerpVectors
  private camLerpAlpha = 1.0;
  private camStartPos = new THREE.Vector3(34, 23, 38);
  private camStartLook = new THREE.Vector3(0, 0, 3);
  private camTargetPos = new THREE.Vector3(34, 23, 38);
  private camTargetLook = new THREE.Vector3(0, 0, 3);

  private onSelectCallback: ((meta: InspectableMetadata, frame: TelemetryFrame | null) => void) | null =
    null;

  private readonly NUM_LAYERS = 8;
  private readonly NUM_EXPERTS = 4; // MiniMind2-MoE: 4 routed + 1 shared
  private readonly LAYER_SPACING = 6.0;

  constructor(container: HTMLElement, labelsContainer: HTMLElement) {
    this.labelsContainer = labelsContainer;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060a12);
    this.scene.fog = new THREE.FogExp2(0x060a12, 0.0082);

    this.camera = new THREE.PerspectiveCamera(
      46,
      window.innerWidth / window.innerHeight,
      0.1,
      350
    );
    this.camera.position.set(34, 23, 38);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxDistance = 140;
    this.controls.minDistance = 8;
    this.controls.target.set(0, 0, 3);
    this.controls.update();

    // Stop automatic camera fly-to if user drags the canvas manually
    this.controls.addEventListener('start', () => {
      this.camLerpAlpha = 1.0;
    });

    this.setupLightsAndGrid();
    this.buildArchitectureGeometry();
    this.buildWavefrontObjects();

    window.addEventListener('resize', () => this.onResize());
    this.renderer.domElement.addEventListener('pointerdown', (e) => this.onPointerDown(e));

    this.animate();
  }

  public onNodeSelected(
    cb: (meta: InspectableMetadata, frame: TelemetryFrame | null) => void
  ): void {
    this.onSelectCallback = cb;
  }

  public resetCamera(): void {
    this.flyCameraTo(new THREE.Vector3(34, 23, 38), new THREE.Vector3(0, 0, 3));
  }

  public focusStageCamera(stage: TrainingStage): void {
    switch (stage) {
      case 'pretrain':
      case 'sft':
        this.flyCameraTo(new THREE.Vector3(33, 22, 36), new THREE.Vector3(0, 0, 2));
        break;
      case 'lora':
        this.flyCameraTo(new THREE.Vector3(36, 20, 34), new THREE.Vector3(3, 0, 2));
        break;
      case 'rl_dpo':
        this.flyCameraTo(new THREE.Vector3(-34, 24, 46), new THREE.Vector3(-2, 0, 8));
        break;
      case 'rl_grpo':
        this.flyCameraTo(new THREE.Vector3(30, 22, 48), new THREE.Vector3(0, 1, 10));
        break;
    }
  }

  public flyCameraToCoords(pos: [number, number, number], lookAt: [number, number, number]): void {
    this.flyCameraTo(new THREE.Vector3(...pos), new THREE.Vector3(...lookAt));
  }

  /**
   * Story Mode camera: fly to explicit coordinates and, if the story is also switching stage,
   * skip the automatic per-stage camera when that stage's first frame arrives. Without this,
   * whichever of the two camera moves happens last wins (a race against the WebSocket reply).
   * Call this BEFORE sending SET_STAGE so a synchronous local-sim frame is covered too.
   */
  public flyCameraForStory(
    pos: [number, number, number],
    lookAt: [number, number, number],
    stage?: TrainingStage
  ): void {
    this.suppressAutoFocusForStage = stage && stage !== this.lastStage ? stage : null;
    this.flyCameraToCoords(pos, lookAt);
  }

  private flyCameraTo(pos: THREE.Vector3, lookAt: THREE.Vector3): void {
    this.camStartPos.copy(this.camera.position);
    this.camStartLook.copy(this.controls.target);
    this.camTargetPos.copy(pos);
    this.camTargetLook.copy(lookAt);
    this.camLerpAlpha = 0.0;
  }

  private setupLightsAndGrid(): void {
    const ambient = new THREE.AmbientLight(0xdbeafe, 0.85);
    this.scene.add(ambient);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.45);
    dirLight1.position.set(28, 42, 22);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf43f5e, 0.95);
    dirLight2.position.set(-28, 22, -25);
    this.scene.add(dirLight2);

    const grid = new THREE.GridHelper(120, 48, 0x1e3a8a, 0x0f172a);
    grid.position.y = -6.5;
    this.scene.add(grid);
  }

  private createLabel(
    text: string,
    worldPos: THREE.Vector3,
    opts?: { isLockBadge?: boolean; isRopeLabel?: boolean }
  ): void {
    const el = document.createElement('div');
    el.className = opts?.isLockBadge ? 'scene-label lock-badge' : 'scene-label';
    el.textContent = text;
    if (opts?.isLockBadge) {
      el.style.display = 'none';
    }
    this.labelsContainer.appendChild(el);
    this.labels.push({
      el,
      worldPos,
      isLockBadge: opts?.isLockBadge,
      isRopeLabel: opts?.isRopeLabel,
    });
  }

  private buildArchitectureGeometry(): void {
    const startZ = -((this.NUM_LAYERS - 1) * this.LAYER_SPACING) / 2;

    // 1. Entry Embedding + RoPE Phase Dial (angle depends purely on token position m!)
    this.ropeGroup = new THREE.Group();
    this.ropeGroup.position.set(0, 0, startZ - 5.2);

    const ropeGeo = new THREE.TorusGeometry(3.1, 0.22, 16, 48);
    const ropeMat = new THREE.MeshStandardMaterial({
      color: 0x22d3ee,
      emissive: 0x0891b2,
      emissiveIntensity: 0.7,
      metalness: 0.5,
      roughness: 0.2,
    });
    const ropeTorus = new THREE.Mesh(ropeGeo, ropeMat);
    this.ropeGroup.add(ropeTorus);

    // 10 position tick marks (m = 0 .. 9) around the RoPE ring
    for (let m = 0; m < 10; m++) {
      const a = m * 0.62;
      const tick = new THREE.Mesh(
        new THREE.BoxGeometry(0.18, 0.55, 0.28),
        new THREE.MeshStandardMaterial({ color: 0x7dd3fc, emissive: 0x0284c7, emissiveIntensity: 0.6 })
      );
      tick.position.set(Math.cos(a) * 3.1, Math.sin(a) * 3.1, 0);
      tick.rotation.z = a + Math.PI / 2;
      this.ropeGroup.add(tick);
    }

    const handGroup = new THREE.Group();
    handGroup.name = 'ropeHandGroup';
    const shaft = new THREE.Mesh(
      new THREE.BoxGeometry(2.8, 0.22, 0.22),
      new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xd97706, emissiveIntensity: 0.9 })
    );
    shaft.position.x = 1.4;
    const tipOrb = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0xfef08a, emissive: 0xf59e0b, emissiveIntensity: 1.0 })
    );
    tipOrb.position.x = 2.95;
    handGroup.add(shaft, tipOrb);
    this.ropeGroup.add(handGroup);

    this.ropeGroup.userData.inspect = {
      title: 'Bảng Từ vựng (6,400 từ) + Kim vị trí RoPE (Góc = m × θ)',
      layerIdx: 0,
      kind: 'embedding',
      shape: '[vocab_size=6400, hidden_dim=512/640]',
      simpleDesc:
        'Giống cuốn từ điển rút gọn 6.400 từ cốt lõi. Kim đồng hồ RoPE chỉ phụ thuộc vào vị trí đứng của từ trong câu (từ thứ 1, thứ 2, thứ 3...), không đổi theo bước train.',
      technicalDesc:
        'Thu gọn vocab_size xuống 6,400 giảm >75% FLOPs ở lớp Embedding & LM Head. RoPE xoay góc pha cố định m·θ_j theo vị trí từ m trong câu.',
    } satisfies InspectableMetadata;

    this.scene.add(this.ropeGroup);
    this.createLabel('Từ điển (6400) + RoPE (m=0 → 0.00 rad)', new THREE.Vector3(0, 4.3, startZ - 5.2), {
      isRopeLabel: true,
    });

    // 2. Build 8 Transformer Layers
    for (let i = 0; i < this.NUM_LAYERS; i++) {
      const z = startZ + i * this.LAYER_SPACING;
      const group = new THREE.Group();
      group.position.set(0, 0, z);

      // GQA Attention Block (8 Query Heads -> 2 KV Heads)
      const gqaGeo = new THREE.BoxGeometry(3.4, 2.1, 1.1);
      const gqaMat = new THREE.MeshPhysicalMaterial({
        color: 0x3b82f6,
        emissive: 0x1d4ed8,
        emissiveIntensity: 0.35,
        roughness: 0.25,
        metalness: 0.2,
        transparent: true,
        opacity: 0.92,
      });
      const gqaMesh = new THREE.Mesh(gqaGeo, gqaMat);
      gqaMesh.position.set(0, -1.5, -1.1);
      gqaMesh.userData.inspect = {
        title: `Tầng ${i} — GQA Attention (8 Q dùng chung 2 KV)`,
        layerIdx: i,
        kind: 'gqa',
        shape: 'Q: [8 heads] | K,V: [2 heads] (Tiết kiệm 75% KV Cache)',
        simpleDesc:
          'Ẩn dụ: Thay vì 8 nhân viên mỗi người giữ 1 tủ hồ sơ riêng, cả 8 người cùng tra cứu chung 2 tủ hồ sơ (KV Cache) — tiết kiệm 75% bộ nhớ khi mô hình sinh câu trả lời.',
        technicalDesc:
          'Grouped-Query Attention (8Q : 2KV) giảm 75% dung lượng KV Cache khi suy luận (Inference / Rollout GRPO) và giảm tham số chiếu W_K, W_V.',
      } satisfies InspectableMetadata;
      group.add(gqaMesh);

      const kvGeo = new THREE.BoxGeometry(0.9, 1.7, 0.9);
      const kvMat = new THREE.MeshPhysicalMaterial({
        color: 0x06b6d4,
        emissive: 0x0891b2,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.92,
      });
      const kvCacheMesh = new THREE.Mesh(kvGeo, kvMat);
      kvCacheMesh.position.set(-2.5, -1.5, -1.1);
      group.add(kvCacheMesh);

      // Dense SwiGLU FFN Block (MiniMind2-Small: 25.8M, hidden=512)
      const denseGeo = new THREE.BoxGeometry(4.0, 2.5, 1.35);
      const denseMat = new THREE.MeshPhysicalMaterial({
        color: 0x6366f1,
        emissive: 0x3730a3,
        emissiveIntensity: 0.4,
        roughness: 0.25,
        transparent: true,
        opacity: 0.92,
      });
      const denseFfnMesh = new THREE.Mesh(denseGeo, denseMat);
      denseFfnMesh.position.set(0, 1.35, 0.7);
      denseFfnMesh.userData.inspect = {
        title: `Tầng ${i} — Khối suy nghĩ nguyên khối (Dense FFN 25.8M)`,
        layerIdx: i,
        kind: 'dense_ffn',
        shape: '[512 → 1408 → 512] (MiniMind2-Small)',
        simpleDesc:
          'Mạng Dense (25.8M núm vặn): Mọi từ đi qua đều huy động toàn bộ 100% nơ-ron của khối này làm việc.',
        technicalDesc:
          'Khối Feed-Forward SwiGLU trong MiniMind2-Small (hidden_size=512, 8 layers, tổng 25.8M tham số). 100% tham số kích hoạt cho mỗi token.',
      } satisfies InspectableMetadata;
      group.add(denseFfnMesh);

      // MoE Group (Router + 1 Shared Expert + 4 Routed Experts = 145M total / 45.2M active)
      const moeGroup = new THREE.Group();
      moeGroup.position.set(0, 1.15, 0.7);

      const routerGeo = new THREE.SphereGeometry(0.6, 20, 20);
      const routerMat = new THREE.MeshPhysicalMaterial({
        color: 0xf59e0b,
        emissive: 0xd97706,
        emissiveIntensity: 0.85,
        transparent: true,
        opacity: 0.95,
      });
      const routerMesh = new THREE.Mesh(routerGeo, routerMat);
      routerMesh.position.set(0, -0.5, -0.85);
      routerMesh.userData.inspect = {
        title: `Tầng ${i} — Bàn Lễ tân Phân luồng (MoE Router)`,
        layerIdx: i,
        kind: 'moe_router',
        shape: '[hidden_dim=640 → Top-2 / 4 Routed Experts]',
        simpleDesc:
          'Ẩn dụ: Giống quầy lễ tân bệnh viện — nhìn mỗi từ đi vào rồi chỉ định đúng 2 bác sĩ chuyên khoa phù hợp nhất xử lý, cộng với 1 bác sĩ đa khoa luôn trực.',
        technicalDesc:
          'Gating Network chọn Top-2/4 Routed Experts cho mỗi token trong MiniMind2-MoE (145M tổng tham số, chỉ kích hoạt ~45.2M mỗi token).',
      } satisfies InspectableMetadata;
      moeGroup.add(routerMesh);

      const sharedGeo = new THREE.CylinderGeometry(0.44, 0.44, 1.75, 16);
      const sharedMat = new THREE.MeshPhysicalMaterial({
        color: 0x10b981,
        emissive: 0x059669,
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.95,
      });
      const sharedExpertMesh = new THREE.Mesh(sharedGeo, sharedMat);
      sharedExpertMesh.position.set(0, 0.75, 0);
      moeGroup.add(sharedExpertMesh);

      const routedExpertMeshes: THREE.Mesh[] = [];
      const expertLoadBars: THREE.Mesh[] = [];
      const routerBeams: THREE.Line[] = [];

      for (let e = 0; e < this.NUM_EXPERTS; e++) {
        const angle = ((e / (this.NUM_EXPERTS - 1)) - 0.5) * Math.PI * 0.85;
        const radius = 3.5;
        const ex = Math.sin(angle) * radius;
        const ey = Math.cos(angle) * 1.65 + 0.55;

        const expGeo = new THREE.BoxGeometry(0.95, 1.25, 0.95);
        const expMat = new THREE.MeshPhysicalMaterial({
          color: 0x334155,
          emissive: 0x0f172a,
          emissiveIntensity: 0.2,
          roughness: 0.3,
          transparent: true,
          opacity: 0.9,
        });
        const expMesh = new THREE.Mesh(expGeo, expMat);
        expMesh.position.set(ex, ey, 0.3);
        expMesh.userData.inspect = {
          title: `Tầng ${i} — Bác sĩ Chuyên khoa E${e} (Routed Expert)`,
          layerIdx: i,
          kind: 'moe_expert',
          shape: '[640 → 1408 → 640]',
          simpleDesc: `Bác sĩ chuyên khoa E${e}: Chỉ bật đèn sáng làm việc và được cập nhật kinh nghiệm khi nằm trong Top-2 của từ đang xử lý.`,
          technicalDesc: `Routed Expert E${e} trong MiniMind2-MoE. Kích hoạt thưa Top-2 và cân bằng tải qua hệ số phạt Auxiliary Loss.`,
        } satisfies InspectableMetadata;
        moeGroup.add(expMesh);
        routedExpertMeshes.push(expMesh);

        const barGeo = new THREE.BoxGeometry(0.32, 0.95, 0.32);
        const barMat = new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x0284c7,
          emissiveIntensity: 0.6,
        });
        const barMesh = new THREE.Mesh(barGeo, barMat);
        barMesh.position.set(ex, ey + 1.1, 0.3);
        moeGroup.add(barMesh);
        expertLoadBars.push(barMesh);

        const pts = [routerMesh.position.clone(), expMesh.position.clone()];
        const beamGeo = new THREE.BufferGeometry().setFromPoints(pts);
        const beamMat = new THREE.LineBasicMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.15,
        });
        const beamLine = new THREE.Line(beamGeo, beamMat);
        moeGroup.add(beamLine);
        routerBeams.push(beamLine);
      }

      group.add(moeGroup);

      // LoRA Bypass Hourglass Adapter on the right side (X = +5.3)
      const loraGroup = new THREE.Group();
      loraGroup.position.set(5.3, 0, -0.2);

      const loraDownGeo = new THREE.ConeGeometry(0.75, 1.45, 16);
      const loraDownMat = new THREE.MeshStandardMaterial({
        color: 0xe879f9,
        emissive: 0xc026d3,
        emissiveIntensity: 0.85,
      });
      const loraDownMesh = new THREE.Mesh(loraDownGeo, loraDownMat);
      loraDownMesh.rotation.x = Math.PI;
      loraDownMesh.position.y = 0.75;

      const loraUpGeo = new THREE.ConeGeometry(0.75, 1.45, 16);
      const loraUpMat = new THREE.MeshStandardMaterial({
        color: 0xf43f5e,
        emissive: 0xe11d48,
        emissiveIntensity: 0.85,
      });
      const loraUpMesh = new THREE.Mesh(loraUpGeo, loraUpMat);
      loraUpMesh.position.y = -0.75;

      const loraHitbox = new THREE.Mesh(
        new THREE.BoxGeometry(1.6, 3.0, 1.6),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      loraHitbox.userData.inspect = {
        title: `Tầng ${i} — Giấy ghi chú LoRA (A × B, Rank-8)`,
        layerIdx: i,
        kind: 'lora_adapter',
        shape: 'q_proj + o_proj (d×d) | Dense: 0.13M (0.51%) | MoE: 0.16M (0.11%)',
        simpleDesc:
          'Ẩn dụ: Cuốn sách gốc bị khóa 🔒 (sóng đạo hàm mờ vẫn đi xuyên qua để truyền thông tin nhưng không sửa sách gốc), chỉ có tờ giấy ghi chú màu tím ở lề phải được cập nhật!',
        technicalDesc:
          'Đóng băng W_0, chỉ lưu Gradients (2B) và AdamW states (12B: FP32 master + m + v) cho A, B (r=8). Đạo hàm kích hoạt vẫn lan truyền qua mạng gốc nhưng chỉ cập nhật nhánh A × B.',
      } satisfies InspectableMetadata;

      loraGroup.add(loraDownMesh, loraUpMesh, loraHitbox);
      loraGroup.visible = false;
      group.add(loraGroup);

      // Frozen Reference Ghost Block (RL DPO left lane)
      const refGeo = new THREE.BoxGeometry(2.5, 2.1, 1.15);
      const refMat = new THREE.MeshPhysicalMaterial({
        color: 0x7dd3fc,
        emissive: 0x0284c7,
        emissiveIntensity: 0.25,
        transparent: true,
        opacity: 0.3,
      });
      const refGhostMesh = new THREE.Mesh(refGeo, refMat);
      refGhostMesh.position.set(-7.2, 0, 0);
      refGhostMesh.visible = false;
      refGhostMesh.userData.inspect = {
        title: `Tầng ${i} — Bản sao Tham chiếu Đóng băng (Reference Model π_ref)`,
        layerIdx: i,
        kind: 'rl_node',
        shape: 'Frozen Copy π_ref (Tách riêng trên thanh VRAM)',
        simpleDesc:
          'Ẩn dụ: Bản gốc được giữ nguyên làm mốc chuẩn để đảm bảo trong lúc học cách trả lời hay hơn, mô hình không bị "học lệch" hay quên tiếng Việt tự nhiên.',
        technicalDesc:
          'Mô hình tham chiếu đóng băng π_ref (chiếm 2 bytes/param riêng biệt trong VRAM) dùng để tính log-ratio và giữ ràng buộc phân kỳ KL.',
      } satisfies InspectableMetadata;
      group.add(refGhostMesh);

      const klGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-5.9, 0, 0),
        new THREE.Vector3(-1.8, -1.2, -1.0),
      ]);
      const klMat = new THREE.LineDashedMaterial({
        color: 0x38bdf8,
        dashSize: 0.4,
        gapSize: 0.2,
      });
      const klBeam = new THREE.Line(klGeo, klMat);
      klBeam.computeLineDistances();
      klBeam.visible = false;
      group.add(klBeam);

      this.scene.add(group);
      this.createLabel(`L${i}`, new THREE.Vector3(-4.5, -2.5, z));
      this.createLabel('🔒 Đóng băng', new THREE.Vector3(0, 3.6, z), { isLockBadge: true });

      this.layerNodes.push({
        group,
        zPos: z,
        gqaMesh,
        kvCacheMesh,
        denseFfnMesh,
        moeGroup,
        routerMesh,
        sharedExpertMesh,
        routedExpertMeshes,
        expertLoadBars,
        routerBeams,
        loraGroup,
        loraDownMesh,
        loraUpMesh,
        refGhostMesh,
        klBeam,
      });
    }

    const endZ = startZ + (this.NUM_LAYERS - 1) * this.LAYER_SPACING + 5.2;

    // 3. Output LM Head & Loss Gate
    const gateGeo = new THREE.RingGeometry(1.5, 3.2, 32);
    const gateMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x059669,
      emissiveIntensity: 0.65,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.82,
    });
    this.lossMaskGate = new THREE.Mesh(gateGeo, gateMat);
    this.lossMaskGate.position.set(0, 0, endZ);
    this.scene.add(this.lossMaskGate);
    this.createLabel('Đầu ra Đoán chữ & Chấm Loss', new THREE.Vector3(0, 4.1, endZ));

    const burstGeo = new THREE.RingGeometry(1.8, 2.3, 32);
    const burstMat = new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0,
    });
    this.lossBurstRing = new THREE.Mesh(burstGeo, burstMat);
    this.lossBurstRing.position.set(0, 0, endZ + 0.3);
    this.scene.add(this.lossBurstRing);

    // 4. DPO Dual Trajectory Arena (y_win vs y_lose)
    this.dpoArenaGroup = new THREE.Group();
    this.dpoArenaGroup.position.set(0, 0, endZ + 4.5);
    const winSphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.05, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0x34d399, emissive: 0x059669, emissiveIntensity: 0.85 })
    );
    winSphere.position.set(2.8, 1.3, 0);
    const loseSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.9, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0xf43f5e, emissive: 0xbe123c, emissiveIntensity: 0.85 })
    );
    loseSphere.position.set(-2.8, -1.2, 0);
    this.dpoArenaGroup.add(winSphere, loseSphere);
    this.dpoArenaGroup.visible = false;
    this.scene.add(this.dpoArenaGroup);

    // 5. GRPO Group Relative Policy Optimization Fan-out (G=4 branches)
    this.grpoArenaGroup = new THREE.Group();
    this.grpoArenaGroup.position.set(0, 0, endZ + 4.5);
    for (let g = 0; g < 4; g++) {
      const branchMesh = new THREE.Mesh(
        new THREE.BoxGeometry(1.15, 1.15, 1.15),
        new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x0284c7,
          emissiveIntensity: 0.75,
        })
      );
      branchMesh.position.set((g - 1.5) * 2.5, 1.2, 0);
      branchMesh.userData.inspect = {
        title: `GRPO Bài làm thử số ${g + 1} (Không cần Mô hình Giám khảo)`,
        layerIdx: 7,
        kind: 'rl_node',
        shape: 'Điểm tương đối A_i = (r_i - Trung_bình) / Độ_lệch',
        simpleDesc:
          'Ẩn dụ: Mô hình tự viết nháp 4 bài trả lời khác nhau rồi lấy điểm trung bình của cả 4 bài làm mốc so sánh. Bài nào cao hơn trung bình được thưởng (xanh), thấp hơn bị phạt (đỏ).',
        technicalDesc:
          'GRPO lấy mẫu nhóm G=4 hoàn thành (o_1..o_4) và chuẩn hóa phần thưởng trong nhóm làm baseline, loại bỏ hoàn toàn mạng Critic.',
      } satisfies InspectableMetadata;
      this.grpoArenaGroup.add(branchMesh);
      this.grpoBranches.push(branchMesh);
    }
    this.grpoArenaGroup.visible = false;
    this.scene.add(this.grpoArenaGroup);
  }

  private buildWavefrontObjects(): void {
    // 10 parallel token packets representing the full sentence batch
    for (let i = 0; i < 10; i++) {
      const pktGeo = new THREE.SphereGeometry(0.32, 14, 14);
      const pktMat = new THREE.MeshStandardMaterial({
        color: 0x22d3ee,
        emissive: 0x0891b2,
        emissiveIntensity: 0.95,
      });
      const pkt = new THREE.Mesh(pktGeo, pktMat);
      pkt.position.set((i - 4.5) * 0.65, 0, 0);
      this.scene.add(pkt);
      this.tokenPacketMeshes.push(pkt);
    }

    // Primary Backward Gradient Wave Bar (updates active weights)
    const waveGeo = new THREE.BoxGeometry(6.8, 4.2, 0.45);
    const waveMat = new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      transparent: true,
      opacity: 0.45,
    });
    this.backwardWaveBar = new THREE.Mesh(waveGeo, waveMat);
    this.backwardWaveBar.visible = false;
    this.scene.add(this.backwardWaveBar);

    // Secondary Faint Activation-Gradient Wave Bar through Base Model during LoRA!
    // Shows that dL/dx still flows backward through the frozen base network even though dL/dW0 is not computed/applied.
    const baseGradGeo = new THREE.BoxGeometry(6.4, 3.8, 0.35);
    const baseGradMat = new THREE.MeshBasicMaterial({
      color: 0xfda4af,
      transparent: true,
      opacity: 0.16,
    });
    this.baseActivationGradWaveBar = new THREE.Mesh(baseGradGeo, baseGradMat);
    this.baseActivationGradWaveBar.visible = false;
    this.scene.add(this.baseActivationGradWaveBar);
  }

  public updateFromTelemetry(frame: TelemetryFrame): void {
    this.latestFrame = frame;
    this.isPaused = frame.paused;
    this.currentSpeed = frame.speed || 1.0;
    this.stepDurationSec = frame.stepDurationSec || 4.8;

    // Soft-sync continuous 60fps cycle clock with server's cycleSeconds
    if (typeof frame.cycleSeconds === 'number') {
      // Wrap-aware distance: local 0.02s and server 4.75s (cycle 4.8s) are only 0.07s apart,
      // so we must not snap the local clock back into the previous step's backward phase.
      const period = this.stepDurationSec;
      const raw = frame.cycleSeconds - this.localCycleSeconds;
      const wrapped = ((((raw + period / 2) % period) + period) % period) - period / 2;
      const diff = Math.abs(wrapped);
      if (this.isPaused || diff > 0.45) {
        this.localCycleSeconds = frame.cycleSeconds;
      }
    }

    const isMoE = frame.architecture === 'moe';
    const isLora = frame.stage === 'lora';
    const isDpo = frame.stage === 'rl_dpo';
    const isGrpo = frame.stage === 'rl_grpo';

    if (this.lastStage !== frame.stage) {
      this.lastStage = frame.stage;
      if (this.suppressAutoFocusForStage === frame.stage) {
        // Story Mode already flew the camera to a hand-picked view for this stage.
        this.suppressAutoFocusForStage = null;
      } else {
        this.suppressAutoFocusForStage = null;
        this.focusStageCamera(frame.stage);
      }
    }

    for (const lbl of this.labels) {
      if (lbl.isLockBadge) {
        lbl.el.style.display = isLora ? 'block' : 'none';
      }
    }

    this.dpoArenaGroup.visible = isDpo;
    this.grpoArenaGroup.visible = isGrpo;

    if (isGrpo && frame.rl) {
      this.grpoBranches.forEach((mesh, idx) => {
        const adv = frame.rl?.groupAdvantages[idx] ?? 0;
        const mat = mesh.material as THREE.MeshStandardMaterial;
        if (adv >= 0) {
          mat.color.setHex(0x34d399);
          mat.emissive.setHex(0x059669);
        } else {
          mat.color.setHex(0xf43f5e);
          mat.emissive.setHex(0x9f1239);
        }
        mesh.scale.setScalar(1 + Math.max(-0.25, Math.min(0.45, adv * 0.25)));
      });
    }

    const activeToken = frame.currentTokens[frame.activeTokenIdx] || frame.currentTokens[0];

    this.layerNodes.forEach((node, layerIdx) => {
      node.denseFfnMesh.visible = !isMoE;
      node.moeGroup.visible = isMoE;
      node.loraGroup.visible = isLora;
      node.refGhostMesh.visible = isDpo;
      node.klBeam.visible = isDpo;

      const gqaMat = node.gqaMesh.material as THREE.MeshPhysicalMaterial;
      const kvMat = node.kvCacheMesh.material as THREE.MeshPhysicalMaterial;
      const denseMat = node.denseFfnMesh.material as THREE.MeshPhysicalMaterial;
      const routerMat = node.routerMesh.material as THREE.MeshPhysicalMaterial;
      const sharedMat = node.sharedExpertMesh.material as THREE.MeshPhysicalMaterial;

      if (isLora) {
        for (const mat of [gqaMat, kvMat, denseMat, routerMat, sharedMat]) {
          mat.color.setHex(0x7dd3fc);
          mat.emissive.setHex(0x0369a1);
          mat.emissiveIntensity = 0.2;
          mat.opacity = 0.26;
        }
      } else {
        gqaMat.color.setHex(0x3b82f6);
        gqaMat.emissive.setHex(0x1d4ed8);
        gqaMat.emissiveIntensity = 0.35;
        gqaMat.opacity = 0.92;

        kvMat.color.setHex(0x06b6d4);
        kvMat.emissive.setHex(0x0891b2);
        kvMat.emissiveIntensity = 0.5;
        kvMat.opacity = 0.92;

        denseMat.color.setHex(0x6366f1);
        denseMat.emissive.setHex(0x3730a3);
        denseMat.emissiveIntensity = 0.4;
        denseMat.opacity = 0.92;

        routerMat.color.setHex(0xf59e0b);
        routerMat.emissive.setHex(0xd97706);
        routerMat.emissiveIntensity = 0.85;
        routerMat.opacity = 0.95;

        sharedMat.color.setHex(0x10b981);
        sharedMat.emissive.setHex(0x059669);
        sharedMat.emissiveIntensity = 0.6;
        sharedMat.opacity = 0.95;
      }

      const layerTelemetry = frame.layers[layerIdx];
      const activeExperts = activeToken?.activeExperts || [];

      node.routedExpertMeshes.forEach((expMesh, eIdx) => {
        const isTopK = activeExperts.includes(eIdx);
        const expMat = expMesh.material as THREE.MeshPhysicalMaterial;
        if (isLora) {
          expMat.color.setHex(0x7dd3fc);
          expMat.emissive.setHex(0x0369a1);
          expMat.emissiveIntensity = isTopK ? 0.35 : 0.12;
          expMat.opacity = 0.26;
        } else if (isTopK) {
          expMat.color.setHex(0xf59e0b);
          expMat.emissive.setHex(0xd97706);
          expMat.emissiveIntensity = 0.92;
          expMat.opacity = 0.98;
        } else {
          expMat.color.setHex(0x1e293b);
          expMat.emissive.setHex(0x0f172a);
          expMat.emissiveIntensity = 0.15;
          expMat.opacity = 0.52;
        }

        const beamMat = node.routerBeams[eIdx].material as THREE.LineBasicMaterial;
        beamMat.opacity = isTopK ? 0.92 : 0.12;
        beamMat.color.setHex(isLora ? 0x7dd3fc : isTopK ? 0xfbbf24 : 0x38bdf8);

        const loadVal = layerTelemetry?.expertLoads[eIdx] ?? 0.25;
        const loadBar = node.expertLoadBars[eIdx];
        loadBar.scale.y = Math.max(0.25, loadVal * 3.6);
      });
    });
  }

  private onPointerDown(event: PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);

    for (const hit of hits) {
      let obj: THREE.Object3D | null = hit.object;
      while (obj) {
        if (obj.userData?.inspect && this.onSelectCallback) {
          this.onSelectCallback(obj.userData.inspect as InspectableMetadata, this.latestFrame);
          return;
        }
        obj = obj.parent;
      }
    }
  }

  private onResize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate());

    const dt = Math.min(0.05, this.clock.getDelta());

    // 1. Exact 0 -> 1 camera fly-to using lerpVectors + cubic ease-out (reaches 100% of target!)
    if (this.camLerpAlpha < 1.0) {
      this.camLerpAlpha = Math.min(1.0, this.camLerpAlpha + dt / 0.85);
      const ease = 1 - Math.pow(1 - this.camLerpAlpha, 3);
      this.camera.position.lerpVectors(this.camStartPos, this.camTargetPos, ease);
      this.controls.target.lerpVectors(this.camStartLook, this.camTargetLook, ease);
    }

    // 2. Advance local 60fps cycle clock smoothly between WebSocket ticks
    if (!this.isPaused) {
      this.localCycleSeconds =
        (this.localCycleSeconds + dt * this.currentSpeed) % this.stepDurationSec;
    }

    const frame = this.latestFrame;
    const startZ = -((this.NUM_LAYERS - 1) * this.LAYER_SPACING) / 2 - 4.5;
    const endZ = -startZ + 0.5;

    if (frame) {
      const { phase, phaseProgress: p } = MiniMindSimulator.computePhaseFromCycle(
        this.localCycleSeconds,
        this.stepDurationSec
      );
      const isLora = frame.stage === 'lora';
      const isMoE = frame.architecture === 'moe';

      // RoPE clock hand points to the position index m (0..9) of the token currently being scanned
      const posIdx = Math.min(
        frame.currentTokens.length - 1,
        Math.floor(p * frame.currentTokens.length)
      );
      const currentPosToken = frame.currentTokens[posIdx];
      if (currentPosToken) {
        const handGroup = this.ropeGroup.getObjectByName('ropeHandGroup');
        if (handGroup) {
          handGroup.rotation.z = currentPosToken.ropeAngleRad;
        }
        for (const lbl of this.labels) {
          if (lbl.isRopeLabel) {
            lbl.el.textContent = `Từ điển (6400) + RoPE (vị trí m=${currentPosToken.positionIdx} → ${currentPosToken.ropeAngleRad} rad)`;
          }
        }
      }

      // PHASE 1: FORWARD (Smooth 60fps parallel token sweep)
      if (phase === 'forward') {
        const currentZ = startZ + (endZ - startZ) * p;
        this.tokenPacketMeshes.forEach((pkt, idx) => {
          pkt.visible = true;
          pkt.scale.setScalar(1);
          const tok = frame.currentTokens[idx];
          const isMaskedPrompt = Boolean(tok?.lossMasked);
          const mat = pkt.material as THREE.MeshStandardMaterial;
          mat.color.setHex(isMaskedPrompt ? 0x94a3b8 : 0x22d3ee);
          mat.emissive.setHex(isMaskedPrompt ? 0x475569 : 0x0891b2);
          pkt.position.set((idx - 4.5) * 0.65, Math.sin(p * Math.PI * 4 + idx) * 0.42, currentZ);
        });
        this.lossBurstRing.visible = false;
        this.backwardWaveBar.visible = false;
        this.baseActivationGradWaveBar.visible = false;
        this.resetLayerPulses();
      }

      // PHASE 2: LOSS COMPUTE
      else if (phase === 'loss_compute') {
        this.tokenPacketMeshes.forEach((pkt, idx) => {
          pkt.visible = true;
          const tok = frame.currentTokens[idx];
          const isMaskedPrompt = Boolean(tok?.lossMasked);
          const mat = pkt.material as THREE.MeshStandardMaterial;
          if (isMaskedPrompt) {
            mat.color.setHex(0x475569);
            mat.emissive.setHex(0x1e293b);
            pkt.scale.setScalar(0.75);
          } else {
            mat.color.setHex(0x34d399);
            mat.emissive.setHex(0x059669);
            pkt.scale.setScalar(1.25 + Math.sin(p * Math.PI) * 0.3);
          }
          pkt.position.set((idx - 4.5) * 0.65, 0, endZ);
        });

        this.lossBurstRing.visible = true;
        const burstScale = 1 + p * 1.45;
        this.lossBurstRing.scale.set(burstScale, burstScale, 1);
        const burstMat = this.lossBurstRing.material as THREE.MeshBasicMaterial;
        burstMat.opacity = (1 - p) * 0.85;
        this.backwardWaveBar.visible = false;
        this.baseActivationGradWaveBar.visible = false;
        this.resetLayerPulses();
      }

      // PHASE 3: BACKWARD & WEIGHT UPDATE
      else if (phase === 'backward') {
        this.tokenPacketMeshes.forEach((pkt) => {
          pkt.visible = false;
          pkt.scale.setScalar(1);
        });
        this.lossBurstRing.visible = false;

        const gradZ = endZ - (endZ - startZ) * p;
        this.backwardWaveBar.visible = true;
        const waveMat = this.backwardWaveBar.material as THREE.MeshBasicMaterial;

        if (isLora) {
          // 1) Bright magenta weight-update wave along the right-hand LoRA A × B bypass lane (X = +5.3)
          this.backwardWaveBar.position.set(5.3, 0, gradZ);
          this.backwardWaveBar.scale.set(0.38, 0.88, 1);
          waveMat.color.setHex(0xe879f9);
          waveMat.opacity = 0.72;

          // 2) Faint activation-gradient wave still flowing through the frozen base network (X = 0)
          this.baseActivationGradWaveBar.visible = true;
          this.baseActivationGradWaveBar.position.set(0, 0, gradZ);
        } else {
          this.backwardWaveBar.position.set(0, 0, gradZ);
          this.backwardWaveBar.scale.set(1, 1, 1);
          waveMat.color.setHex(0xf43f5e);
          waveMat.opacity = 0.45;
          this.baseActivationGradWaveBar.visible = false;
        }

        const activeExperts =
          (frame.currentTokens[frame.activeTokenIdx] || frame.currentTokens[0])?.activeExperts || [0, 1];

        // Pulse every trainable block in each layer as the backward wave crosses its Z position!
        this.layerNodes.forEach((node) => {
          const dist = Math.abs(node.zPos - gradZ);
          const pulse = dist < 3.4 ? 1 + (1 - dist / 3.4) * 0.26 : 1.0;

          if (isLora) {
            // Only the LoRA A × B hourglass cones pulse; all frozen base blocks stay locked at scale 1.0
            node.loraGroup.scale.setScalar(pulse * 1.15);
            node.gqaMesh.scale.setScalar(1.0);
            node.kvCacheMesh.scale.setScalar(1.0);
            node.denseFfnMesh.scale.setScalar(1.0);
            node.routerMesh.scale.setScalar(1.0);
            node.sharedExpertMesh.scale.setScalar(1.0);
            node.routedExpertMeshes.forEach((m) => m.scale.setScalar(1.0));
          } else {
            node.loraGroup.scale.setScalar(1.0);
            node.gqaMesh.scale.setScalar(pulse);
            node.kvCacheMesh.scale.setScalar(pulse);
            if (!isMoE) {
              node.denseFfnMesh.scale.setScalar(pulse);
            } else {
              // In MoE mode, pulse the Router, Shared Expert, and Top-2 active Routed Experts!
              node.routerMesh.scale.setScalar(pulse * 1.08);
              node.sharedExpertMesh.scale.setScalar(pulse);
              node.routedExpertMeshes.forEach((expMesh, eIdx) => {
                expMesh.scale.setScalar(activeExperts.includes(eIdx) ? pulse * 1.12 : 1.0);
              });
            }
          }
        });
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.updateFloatingLabels();
  }

  private resetLayerPulses(): void {
    for (const node of this.layerNodes) {
      node.loraGroup.scale.setScalar(1.0);
      node.gqaMesh.scale.setScalar(1.0);
      node.kvCacheMesh.scale.setScalar(1.0);
      node.denseFfnMesh.scale.setScalar(1.0);
      node.routerMesh.scale.setScalar(1.0);
      node.sharedExpertMesh.scale.setScalar(1.0);
      for (const expMesh of node.routedExpertMeshes) {
        expMesh.scale.setScalar(1.0);
      }
    }
  }

  private updateFloatingLabels(): void {
    const widthHalf = window.innerWidth / 2;
    const heightHalf = window.innerHeight / 2;

    for (const lbl of this.labels) {
      if (lbl.el.style.display === 'none') continue;
      const pos = lbl.worldPos.clone().project(this.camera);
      if (pos.z > 1) {
        lbl.el.style.opacity = '0';
        continue;
      }
      const x = pos.x * widthHalf + widthHalf;
      const y = -(pos.y * heightHalf) + heightHalf;
      lbl.el.style.opacity = '1';
      lbl.el.style.left = `${x}px`;
      lbl.el.style.top = `${y}px`;
    }
  }
}
