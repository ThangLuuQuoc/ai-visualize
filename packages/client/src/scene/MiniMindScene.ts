import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TelemetryFrame } from '@minimind-viz/shared';

export interface InspectableMetadata {
  title: string;
  layerIdx: number;
  kind: 'embedding' | 'gqa' | 'dense_ffn' | 'moe_router' | 'moe_expert' | 'lora_adapter' | 'rl_node';
  description: string;
  shape: string;
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
}

export class MiniMindScene {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();

  private layerNodes: Layer3DNodes[] = [];
  private ropeRing!: THREE.Mesh;
  private lossMaskGate!: THREE.Mesh;
  private dpoArenaGroup!: THREE.Group;
  private grpoArenaGroup!: THREE.Group;
  private grpoBranches: THREE.Mesh[] = [];

  private forwardParticles!: THREE.Points;
  private backwardParticles!: THREE.Points;
  private forwardPositions!: Float32Array;
  private backwardPositions!: Float32Array;

  private labels: FloatingLabel[] = [];
  private labelsContainer: HTMLElement;
  private latestFrame: TelemetryFrame | null = null;
  private onSelectCallback: ((meta: InspectableMetadata, frame: TelemetryFrame | null) => void) | null =
    null;

  private readonly NUM_LAYERS = 8;
  private readonly NUM_EXPERTS = 6;
  private readonly LAYER_SPACING = 6.2;

  constructor(container: HTMLElement, labelsContainer: HTMLElement) {
    this.labelsContainer = labelsContainer;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060a12);
    this.scene.fog = new THREE.FogExp2(0x060a12, 0.011);

    this.camera = new THREE.PerspectiveCamera(
      48,
      window.innerWidth / window.innerHeight,
      0.1,
      300
    );
    this.resetCamera();

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 120;
    this.controls.minDistance = 10;

    this.setupLightsAndGrid();
    this.buildArchitectureGeometry();
    this.buildParticleSystems();

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
    this.camera.position.set(28, 20, 32);
    this.camera.lookAt(0, 0, 0);
    if (this.controls) {
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
  }

  private setupLightsAndGrid(): void {
    const ambient = new THREE.AmbientLight(0xdbeafe, 0.75);
    this.scene.add(ambient);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.4);
    dirLight1.position.set(25, 40, 20);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf43f5e, 0.9);
    dirLight2.position.set(-25, 20, -25);
    this.scene.add(dirLight2);

    const grid = new THREE.GridHelper(110, 44, 0x1e3a8a, 0x0f172a);
    grid.position.y = -6.5;
    this.scene.add(grid);
  }

  private createLabel(text: string, worldPos: THREE.Vector3): void {
    const el = document.createElement('div');
    el.className = 'scene-label';
    el.textContent = text;
    this.labelsContainer.appendChild(el);
    this.labels.push({ el, worldPos });
  }

  private buildArchitectureGeometry(): void {
    const startZ = -((this.NUM_LAYERS - 1) * this.LAYER_SPACING) / 2;

    // 1. Entry Embedding + RoPE Phase Ring
    const ropeGeo = new THREE.TorusGeometry(3.2, 0.22, 16, 48);
    const ropeMat = new THREE.MeshStandardMaterial({
      color: 0x22d3ee,
      emissive: 0x0891b2,
      emissiveIntensity: 0.7,
      metalness: 0.5,
      roughness: 0.2,
    });
    this.ropeRing = new THREE.Mesh(ropeGeo, ropeMat);
    this.ropeRing.position.set(0, 0, startZ - 5.2);
    this.ropeRing.userData.inspect = {
      title: 'Token Embedding + RoPE (Rotary Position)',
      layerIdx: 0,
      kind: 'embedding',
      shape: '[vocab_size=6400, hidden_dim=768]',
      description:
        'MiniMind thu gọn từ điển xuống 6,400 token giúp giảm 75% kích thước ma trận Embedding & LM Head, kết hợp xoay góc pha phức RoPE.',
    } satisfies InspectableMetadata;
    this.scene.add(this.ropeRing);
    this.createLabel('Embed (6400) + RoPE', new THREE.Vector3(0, 4.2, startZ - 5.2));

    // 2. Build 8 Transformer Layers
    for (let i = 0; i < this.NUM_LAYERS; i++) {
      const z = startZ + i * this.LAYER_SPACING;
      const group = new THREE.Group();
      group.position.set(0, 0, z);

      // GQA Attention Block (8 Query Heads -> 2 KV Heads)
      const gqaGeo = new THREE.BoxGeometry(3.4, 2.2, 1.1);
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
      gqaMesh.position.set(0, -1.5, -1.2);
      gqaMesh.userData.inspect = {
        title: `Layer ${i} — GQA Attention + FlashAttention`,
        layerIdx: i,
        kind: 'gqa',
        shape: 'Q: [8 heads × 96] | K,V: [2 heads × 96]',
        description:
          'Grouped-Query Attention (8Q : 2KV) giảm 75% băng thông bộ nhớ KV Cache, kết hợp FlashAttention hợp nhất phép tính trong SRAM.',
      } satisfies InspectableMetadata;
      group.add(gqaMesh);

      // Compact KV Cache bar next to GQA
      const kvGeo = new THREE.BoxGeometry(0.9, 1.8, 0.9);
      const kvMat = new THREE.MeshStandardMaterial({
        color: 0x06b6d4,
        emissive: 0x0891b2,
        emissiveIntensity: 0.5,
      });
      const kvCacheMesh = new THREE.Mesh(kvGeo, kvMat);
      kvCacheMesh.position.set(-2.5, -1.5, -1.2);
      group.add(kvCacheMesh);

      // Dense SwiGLU FFN Block (Visible in Dense mode)
      const denseGeo = new THREE.BoxGeometry(4.2, 2.6, 1.4);
      const denseMat = new THREE.MeshPhysicalMaterial({
        color: 0x6366f1,
        emissive: 0x3730a3,
        emissiveIntensity: 0.4,
        roughness: 0.25,
        transparent: true,
        opacity: 0.92,
      });
      const denseFfnMesh = new THREE.Mesh(denseGeo, denseMat);
      denseFfnMesh.position.set(0, 1.4, 0.8);
      denseFfnMesh.userData.inspect = {
        title: `Layer ${i} — Dense SwiGLU FFN`,
        layerIdx: i,
        kind: 'dense_ffn',
        shape: '[768 → 2048 → 768]',
        description:
          'Khối Feed-Forward SwiGLU tiêu chuẩn trong biến thể MiniMind Dense (64M tham số). Toàn bộ tham số FFN đều kích hoạt cho mọi token.',
      } satisfies InspectableMetadata;
      group.add(denseFfnMesh);

      // MoE Group (Router + 1 Shared Expert + 6 Routed Experts)
      const moeGroup = new THREE.Group();
      moeGroup.position.set(0, 1.2, 0.8);

      const routerGeo = new THREE.SphereGeometry(0.62, 20, 20);
      const routerMat = new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0xd97706,
        emissiveIntensity: 0.85,
      });
      const routerMesh = new THREE.Mesh(routerGeo, routerMat);
      routerMesh.position.set(0, -0.5, -0.9);
      routerMesh.userData.inspect = {
        title: `Layer ${i} — MoE Gating Router (Top-2 / 6)`,
        layerIdx: i,
        kind: 'moe_router',
        shape: '[hidden_dim=768 → 6 expert logits]',
        description:
          'Cổng Router chấm điểm và chỉ kích hoạt Top-2 Routed Experts cho mỗi token, giữ chi phí tính toán ở mức ~64M active params dù tổng mạng đạt 198M.',
      } satisfies InspectableMetadata;
      moeGroup.add(routerMesh);

      // Shared Expert in center
      const sharedGeo = new THREE.CylinderGeometry(0.45, 0.45, 1.8, 16);
      const sharedMat = new THREE.MeshStandardMaterial({
        color: 0x10b981,
        emissive: 0x059669,
        emissiveIntensity: 0.6,
      });
      const sharedExpertMesh = new THREE.Mesh(sharedGeo, sharedMat);
      sharedExpertMesh.position.set(0, 0.8, 0);
      moeGroup.add(sharedExpertMesh);

      const routedExpertMeshes: THREE.Mesh[] = [];
      const expertLoadBars: THREE.Mesh[] = [];
      const routerBeams: THREE.Line[] = [];

      for (let e = 0; e < this.NUM_EXPERTS; e++) {
        const angle = (e / this.NUM_EXPERTS) * Math.PI - Math.PI / 2;
        const radius = 3.6;
        const ex = Math.sin(angle) * radius;
        const ey = Math.cos(angle) * 1.8 + 0.6;

        const expGeo = new THREE.BoxGeometry(0.92, 1.25, 0.92);
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
          title: `Layer ${i} — Routed Expert E${e}`,
          layerIdx: i,
          kind: 'moe_expert',
          shape: '[768 → 1408 → 768]',
          description: `Chuyên gia định tuyến E${e}. Chỉ tính toán khi nằm trong Top-2 của token hiện tại; được cân bằng tải qua Auxiliary Loss.`,
        } satisfies InspectableMetadata;
        moeGroup.add(expMesh);
        routedExpertMeshes.push(expMesh);

        // 3D Load Bar above each expert
        const barGeo = new THREE.BoxGeometry(0.32, 1.0, 0.32);
        const barMat = new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x0284c7,
          emissiveIntensity: 0.6,
        });
        const barMesh = new THREE.Mesh(barGeo, barMat);
        barMesh.position.set(ex, ey + 1.1, 0.3);
        moeGroup.add(barMesh);
        expertLoadBars.push(barMesh);

        // Beam from Router to Expert
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

      // LoRA Bypass Hourglass Adapter (A: 768->8, B: 8->768) on the right side
      const loraGroup = new THREE.Group();
      loraGroup.position.set(5.4, 0, -0.3);

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
        title: `Layer ${i} — LoRA Rank-8 Adapter (A × B)`,
        layerIdx: i,
        kind: 'lora_adapter',
        shape: 'Matrix A: [768 × 8] → Matrix B: [8 × 768]',
        description:
          'Đóng băng toàn bộ trọng số gốc W0 (0 gradient) và chỉ cập nhật nhánh thắt cổ chai rank r=8 (~1.24M tham số toàn mạng), giảm >55% VRAM.',
      } satisfies InspectableMetadata;

      loraGroup.add(loraDownMesh, loraUpMesh, loraHitbox);
      loraGroup.visible = false;
      group.add(loraGroup);

      // Frozen Reference Ghost Block (Visible in RL DPO mode on left lane)
      const refGeo = new THREE.BoxGeometry(2.6, 2.2, 1.2);
      const refMat = new THREE.MeshPhysicalMaterial({
        color: 0x7dd3fc,
        emissive: 0x0284c7,
        emissiveIntensity: 0.25,
        transparent: true,
        opacity: 0.3,
        wireframe: false,
      });
      const refGhostMesh = new THREE.Mesh(refGeo, refMat);
      refGhostMesh.position.set(-7.5, 0, 0);
      refGhostMesh.visible = false;
      refGhostMesh.userData.inspect = {
        title: `Layer ${i} — Frozen Reference Model (π_ref)`,
        layerIdx: i,
        kind: 'rl_node',
        shape: 'Frozen Copy + KL-Divergence Anchor',
        description:
          'Bản sao đóng băng dùng làm mốc tham chiếu trong DPO để tính tỷ số log-prob và giữ Policy Model không trôi quá xa qua lực kéo KL.',
      } satisfies InspectableMetadata;
      group.add(refGhostMesh);

      const klGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-6.2, 0, 0),
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
      this.createLabel(`L${i}`, new THREE.Vector3(-4.6, -2.5, z));

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

    const endZ = startZ + (this.NUM_LAYERS - 1) * this.LAYER_SPACING + 5.5;

    // 3. Output Loss Masking Shield Gate (SFT / LoRA prompt masking)
    const gateGeo = new THREE.RingGeometry(1.4, 3.3, 32);
    const gateMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x059669,
      emissiveIntensity: 0.65,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.78,
    });
    this.lossMaskGate = new THREE.Mesh(gateGeo, gateMat);
    this.lossMaskGate.position.set(0, 0, endZ);
    this.scene.add(this.lossMaskGate);
    this.createLabel('LM Head & Loss Gate', new THREE.Vector3(0, 4.1, endZ));

    // 4. DPO Dual Trajectory Arena (y_win vs y_lose)
    this.dpoArenaGroup = new THREE.Group();
    this.dpoArenaGroup.position.set(0, 0, endZ + 4.2);
    const winSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0x34d399, emissive: 0x059669, emissiveIntensity: 0.8 })
    );
    winSphere.position.set(2.6, 1.2, 0);
    const loseSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.85, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0xf43f5e, emissive: 0xbe123c, emissiveIntensity: 0.8 })
    );
    loseSphere.position.set(-2.6, -1.2, 0);
    this.dpoArenaGroup.add(winSphere, loseSphere);
    this.dpoArenaGroup.visible = false;
    this.scene.add(this.dpoArenaGroup);

    // 5. GRPO Group Relative Policy Optimization Fan-out (G=4 branches, no Critic)
    this.grpoArenaGroup = new THREE.Group();
    this.grpoArenaGroup.position.set(0, 0, endZ + 4.2);
    for (let g = 0; g < 4; g++) {
      const branchMesh = new THREE.Mesh(
        new THREE.BoxGeometry(1.1, 1.1, 1.1),
        new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x0284c7,
          emissiveIntensity: 0.7,
        })
      );
      branchMesh.position.set((g - 1.5) * 2.4, 1.2, 0);
      branchMesh.userData.inspect = {
        title: `GRPO Group Output o_${g + 1} (No Critic Model)`,
        layerIdx: 7,
        kind: 'rl_node',
        shape: 'Group Baseline Advantage A_i = (r_i - mean(R)) / std(R)',
        description:
          'GRPO lấy mẫu nhóm G=4 câu trả lời và tự chuẩn hóa điểm thưởng tương đối trong nhóm, loại bỏ hoàn toàn mô hình Critic (tiết kiệm ~50% VRAM so với PPO).',
      } satisfies InspectableMetadata;
      this.grpoArenaGroup.add(branchMesh);
      this.grpoBranches.push(branchMesh);
    }
    this.grpoArenaGroup.visible = false;
    this.scene.add(this.grpoArenaGroup);
  }

  private buildParticleSystems(): void {
    const count = 180;
    this.forwardPositions = new Float32Array(count * 3);
    this.backwardPositions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      this.forwardPositions[i * 3] = (Math.random() - 0.5) * 3.5;
      this.forwardPositions[i * 3 + 1] = (Math.random() - 0.5) * 3.0;
      this.forwardPositions[i * 3 + 2] = (Math.random() - 0.5) * 52;

      this.backwardPositions[i * 3] = (Math.random() - 0.5) * 3.5;
      this.backwardPositions[i * 3 + 1] = (Math.random() - 0.5) * 3.0;
      this.backwardPositions[i * 3 + 2] = (Math.random() - 0.5) * 52;
    }

    const fwdGeo = new THREE.BufferGeometry();
    fwdGeo.setAttribute('position', new THREE.BufferAttribute(this.forwardPositions, 3));
    const fwdMat = new THREE.PointsMaterial({
      color: 0x22d3ee,
      size: 0.34,
      transparent: true,
      opacity: 0.85,
    });
    this.forwardParticles = new THREE.Points(fwdGeo, fwdMat);
    this.scene.add(this.forwardParticles);

    const bwdGeo = new THREE.BufferGeometry();
    bwdGeo.setAttribute('position', new THREE.BufferAttribute(this.backwardPositions, 3));
    const bwdMat = new THREE.PointsMaterial({
      color: 0xfb7185,
      size: 0.38,
      transparent: true,
      opacity: 0.85,
    });
    this.backwardParticles = new THREE.Points(bwdGeo, bwdMat);
    this.scene.add(this.backwardParticles);
  }

  public updateFromTelemetry(frame: TelemetryFrame): void {
    this.latestFrame = frame;
    const isMoE = frame.architecture === 'moe';
    const isLora = frame.stage === 'lora';
    const isDpo = frame.stage === 'rl_dpo';
    const isGrpo = frame.stage === 'rl_grpo';

    const activeToken = frame.currentTokens[frame.activeTokenIdx] || frame.currentTokens[0];
    if (activeToken) {
      this.ropeRing.rotation.z = activeToken.ropeAngleRad;
    }

    // Loss Masking Gate visual state
    const gateMat = this.lossMaskGate.material as THREE.MeshStandardMaterial;
    if (activeToken?.lossMasked) {
      gateMat.color.setHex(0xf43f5e);
      gateMat.emissive.setHex(0xbe123c);
    } else {
      gateMat.color.setHex(0x10b981);
      gateMat.emissive.setHex(0x059669);
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
        mesh.scale.setScalar(1 + Math.max(-0.3, Math.min(0.45, adv * 0.25)));
      });
    }

    // Update each of the 8 layers
    this.layerNodes.forEach((node, layerIdx) => {
      node.denseFfnMesh.visible = !isMoE;
      node.moeGroup.visible = isMoE;
      node.loraGroup.visible = isLora;
      node.refGhostMesh.visible = isDpo;
      node.klBeam.visible = isDpo;

      // Frozen Ice-Blue Glass look on Base weights during LoRA
      const gqaMat = node.gqaMesh.material as THREE.MeshPhysicalMaterial;
      const denseMat = node.denseFfnMesh.material as THREE.MeshPhysicalMaterial;
      if (isLora) {
        gqaMat.color.setHex(0x7dd3fc);
        gqaMat.emissive.setHex(0x0369a1);
        gqaMat.opacity = 0.28;
        denseMat.color.setHex(0x7dd3fc);
        denseMat.emissive.setHex(0x0369a1);
        denseMat.opacity = 0.28;
      } else {
        gqaMat.color.setHex(0x3b82f6);
        gqaMat.emissive.setHex(0x1d4ed8);
        gqaMat.opacity = 0.92;
        denseMat.color.setHex(0x6366f1);
        denseMat.emissive.setHex(0x3730a3);
        denseMat.opacity = 0.92;
      }

      // Highlight Top-2 active experts and update 3D load bars
      const layerTelemetry = frame.layers[layerIdx];
      const activeExperts = activeToken?.activeExperts || [];

      node.routedExpertMeshes.forEach((expMesh, eIdx) => {
        const isTopK = activeExperts.includes(eIdx);
        const expMat = expMesh.material as THREE.MeshPhysicalMaterial;
        if (isLora) {
          expMat.color.setHex(isTopK ? 0x38bdf8 : 0x1e293b);
          expMat.opacity = 0.32;
        } else if (isTopK) {
          expMat.color.setHex(0xf59e0b);
          expMat.emissive.setHex(0xd97706);
          expMat.emissiveIntensity = 0.9;
          expMat.opacity = 0.98;
        } else {
          expMat.color.setHex(0x1e293b);
          expMat.emissive.setHex(0x0f172a);
          expMat.emissiveIntensity = 0.15;
          expMat.opacity = 0.55;
        }

        const beamMat = node.routerBeams[eIdx].material as THREE.LineBasicMaterial;
        beamMat.opacity = isTopK ? 0.95 : 0.12;
        beamMat.color.setHex(isTopK ? 0xfbbf24 : 0x38bdf8);

        const loadVal = layerTelemetry?.expertLoads[eIdx] ?? 0.16;
        const loadBar = node.expertLoadBars[eIdx];
        loadBar.scale.y = Math.max(0.2, loadVal * 4.2);
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

    const frame = this.latestFrame;
    const isLora = frame?.stage === 'lora';
    const activeToken = frame?.currentTokens[frame.activeTokenIdx];
    const isMasked = Boolean(activeToken?.lossMasked);

    // Animate forward particles (+Z direction)
    const fwdAttr = this.forwardParticles.geometry.getAttribute(
      'position'
    ) as THREE.BufferAttribute;
    for (let i = 0; i < fwdAttr.count; i++) {
      let z = fwdAttr.getZ(i) + 0.42;
      if (z > 27) z = -27;
      fwdAttr.setZ(i, z);
    }
    fwdAttr.needsUpdate = true;

    // Animate backward gradient particles (-Z direction)
    this.backwardParticles.visible = !isMasked;
    const bwdMat = this.backwardParticles.material as THREE.PointsMaterial;
    bwdMat.color.setHex(isLora ? 0xe879f9 : 0xfb7185);

    const bwdAttr = this.backwardParticles.geometry.getAttribute(
      'position'
    ) as THREE.BufferAttribute;
    for (let i = 0; i < bwdAttr.count; i++) {
      let z = bwdAttr.getZ(i) - 0.45;
      if (z < -27) z = 27;
      bwdAttr.setZ(i, z);
      // In LoRA mode, shift gradient stream to flow exclusively along the X=+5.4 LoRA bypass lane!
      const targetX = isLora
        ? 5.2 + (Math.sin(i + z) * 0.45)
        : ((i % 7) - 3) * 0.9;
      bwdAttr.setX(i, targetX);
    }
    bwdAttr.needsUpdate = true;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.updateFloatingLabels();
  }

  private updateFloatingLabels(): void {
    const widthHalf = window.innerWidth / 2;
    const heightHalf = window.innerHeight / 2;

    for (const lbl of this.labels) {
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
