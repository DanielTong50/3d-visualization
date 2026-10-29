/** Stud-grid source and deterministic compiler output, mirrored from darius.lego. */
export interface LegoBrick {
  id: string;
  part_id: string;
  color: string;
  color_hex: string;
  studs: [number, number];
  height_plates: number;
  position: [number, number, number];
  rotation: 0 | 90 | 180 | 270;
}
export interface LegoAssembly {
  schema_version: string;
  id: string;
  title: string;
  description: string;
  source: { kind: string; [key: string]: unknown };
  baseplate: { studs: [number, number]; color_hex: string; height_mm?: number };
  bricks: LegoBrick[];
  steps: { index: number; title: string; brick_ids: string[] }[];
}
export interface LegoPrimitive {
  kind: string;
  position_mm: number[];
  yaw_deg: number;
  duration_s: number;
  gripper?: 'open' | 'closed' | 'hold';
  force_limit_n?: number | null;
}
export interface LegoPlan {
  assembly: LegoAssembly;
  connections: { id: string; lower: string; upper: string; studs: number[][]; step: number }[];
  steps: {
    index: number;
    brick_id: string;
    dependencies: string[];
    connection_ids: string[];
    source_center_mm?: number[];
    target_center_mm?: number[];
    source_yaw_deg?: number;
    target_yaw_deg?: number;
    primitives: LegoPrimitive[];
  }[];
  validation: { valid: boolean; errors: string[]; warnings: string[] };
  metrics: { bricks: number; connections: number; stud_connections: number; steps: number; primitives: number };
  execution: { status: string; blockers: string[] };
}
export interface LegoSummary {
  id: string;
  title: string;
  description?: string;
  bricks?: number;
  steps?: number;
}
export const BUILD_IDS = ['signal-tower', 'bridge', 'stairway'] as const;
export const STEP_SECONDS = 5.1;
export function timelineDuration(steps: number): number {
  // Keep the range input's maximum on its decimal step grid (6 * 5.1 is not exact in binary).
  return Math.round(steps * STEP_SECONDS * 1000) / 1000;
}
export const PHASES = ['Approach', 'Grasp', 'Transfer', 'Align', 'Press', 'Verify'] as const;
export function phaseAt(fraction: number): number {
  return fraction < 0.18 ? 0 : fraction < 0.32 ? 1 : fraction < 0.64 ? 2 : fraction < 0.8 ? 3 : fraction < 0.94 ? 4 : 5;
}
export function timelineAt(time: number, steps: number): { completed: number; active: number; fraction: number; phase: number } {
  const bounded = Math.max(0, Math.min(Number.isFinite(time) ? time : 0, timelineDuration(steps)));
  const completed = Math.min(steps, Math.floor(bounded / STEP_SECONDS));
  const fraction = completed === steps ? 1 : (bounded % STEP_SECONDS) / STEP_SECONDS;
  return { completed, active: Math.min(completed, Math.max(0, steps - 1)), fraction, phase: phaseAt(fraction) };
}
export function brickFootprint(brick: LegoBrick): [number, number] {
  return brick.rotation % 180 ? [brick.studs[1], brick.studs[0]] : brick.studs;
}
export function instructionUrl(set: string): string {
  if (!/^\d{4,7}$/.test(set)) throw new Error('Enter a set number with 4–7 digits.');
  return `https://www.lego.com/en-us/service/building-instructions/${set}`;
}
/** Exact host allowlist shared with the server's official-asset policy. */
export function isOfficialAssetUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' &&
      ['www.lego.com', 'lego.com', 'www.legocdn.com'].includes(url.hostname) &&
      !url.username &&
      !url.password &&
      (!url.port || url.port === '443')
    );
  } catch {
    return false;
  }
}

/** Convert compiler millimeters to the viewer's 8mm units, retaining Y-up and a centered baseplate. */
export function renderPlacement(
  plan: LegoPlan,
  step: LegoPlan['steps'][number],
  brick: LegoBrick,
  index = 0,
): { target: [number, number, number]; source: [number, number, number]; trayYaw: number } {
  const [bw, bd] = plan.assembly.baseplate.studs;
  const [w, d] = brickFootprint(brick);
  const halfHeight = brick.height_plates * 0.2;
  const point = (center: number[]): [number, number, number] => [
    center[0] / 8 - bw / 2,
    center[1] / 8 - halfHeight,
    center[2] / 8 - bd / 2,
  ];
  return {
    target: step.target_center_mm
      ? point(step.target_center_mm)
      : [
          brick.position[0] + w / 2 - bw / 2,
          brick.position[1] * 0.4 + (plan.assembly.baseplate.height_mm ?? 3.2) / 8,
          brick.position[2] + d / 2 - bd / 2,
        ],
    source: step.source_center_mm ? point(step.source_center_mm) : [12 + (index % 2) * 3.2, 0, -7 + Math.floor(index / 2) * 3.2],
    trayYaw: (((step.source_yaw_deg ?? 0) - (step.target_yaw_deg ?? brick.rotation)) * Math.PI) / 180,
  };
}

export function downloadJson(name: string, data: unknown): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Bound imported artifacts before allocating WebGL meshes. Semantic validation stays in the compiler. */
export function parseCompiledPlan(value: unknown): LegoPlan {
  const fail = () => {
    throw new Error(
      'Import a compiled Darius LEGO plan.json (maximum 128 bricks). Compile source assemblies with darius lego compile first.',
    );
  };
  if (!value || typeof value !== 'object') return fail();
  const p = value as LegoPlan;
  const vec = (v: unknown, n: number): v is number[] =>
    Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number' && Number.isFinite(x) && Math.abs(x) <= 10000);
  const text = (s: unknown) => typeof s === 'string' && s.length > 0 && s.length < 200;
  if (
    !p.assembly ||
    !text(p.assembly.id) ||
    !text(p.assembly.title) ||
    !p.assembly.source ||
    typeof p.assembly.source !== 'object' ||
    !text(p.assembly.source.kind) ||
    !p.assembly.baseplate ||
    !vec(p.assembly.baseplate.studs, 2) ||
    (p.assembly.baseplate.height_mm !== undefined &&
      (typeof p.assembly.baseplate.height_mm !== 'number' ||
        !Number.isFinite(p.assembly.baseplate.height_mm) ||
        p.assembly.baseplate.height_mm <= 0 ||
        p.assembly.baseplate.height_mm > 500)) ||
    p.assembly.baseplate.studs.some((v) => v < 1 || v > 64 || !Number.isInteger(v))
  )
    return fail();
  if (
    !Array.isArray(p.assembly.bricks) ||
    !p.assembly.bricks.length ||
    p.assembly.bricks.length > 128 ||
    !Array.isArray(p.steps) ||
    p.steps.length !== p.assembly.bricks.length ||
    !Array.isArray(p.assembly.steps) ||
    !Array.isArray(p.connections) ||
    p.connections.length > 4096
  )
    return fail();
  const ids = new Set<string>();
  let totalStuds = p.assembly.baseplate.studs[0] * p.assembly.baseplate.studs[1];
  for (const b of p.assembly.bricks) {
    if (
      !b ||
      typeof b !== 'object' ||
      !text(b.id) ||
      ids.has(b.id) ||
      !text(b.part_id) ||
      !text(b.color) ||
      !/^#[0-9a-f]{6}$/i.test(b.color_hex) ||
      !vec(b.studs, 2) ||
      b.studs.some((v) => v < 1 || v > 32 || !Number.isInteger(v)) ||
      !vec(b.position, 3) ||
      !Number.isInteger(b.height_plates) ||
      b.height_plates < 1 ||
      b.height_plates > 32 ||
      ![0, 90, 180, 270].includes(b.rotation)
    )
      return fail();
    totalStuds += b.studs[0] * b.studs[1];
    if (totalStuds > 8192) throw new Error('Imported plans may contain at most 8,192 studs, including the baseplate.');
    ids.add(b.id);
  }
  if (p.steps.some((s) => !s || typeof s !== 'object') || new Set(p.steps.map((s) => s.brick_id)).size !== ids.size) return fail();
  for (const s of p.steps) {
    if (
      !ids.has(s.brick_id) ||
      !Number.isInteger(s.index) ||
      !Array.isArray(s.dependencies) ||
      !s.dependencies.every((id) => ids.has(id)) ||
      !Array.isArray(s.connection_ids) ||
      !s.connection_ids.every(text) ||
      !Array.isArray(s.primitives) ||
      !s.primitives.length ||
      s.primitives.length > 100 ||
      (s.source_center_mm !== undefined && !vec(s.source_center_mm, 3)) ||
      (s.target_center_mm !== undefined && !vec(s.target_center_mm, 3)) ||
      (s.source_yaw_deg !== undefined &&
        (typeof s.source_yaw_deg !== 'number' || !Number.isFinite(s.source_yaw_deg) || Math.abs(s.source_yaw_deg) > 36000)) ||
      (s.target_yaw_deg !== undefined &&
        (typeof s.target_yaw_deg !== 'number' || !Number.isFinite(s.target_yaw_deg) || Math.abs(s.target_yaw_deg) > 36000))
    )
      return fail();
    for (const primitive of s.primitives) {
      if (
        !primitive ||
        typeof primitive !== 'object' ||
        !['move', 'grasp', 'release', 'press', 'verify'].includes(primitive.kind) ||
        !vec(primitive.position_mm, 3) ||
        typeof primitive.duration_s !== 'number' ||
        !Number.isFinite(primitive.duration_s) ||
        primitive.duration_s <= 0 ||
        primitive.duration_s > 3600 ||
        typeof primitive.yaw_deg !== 'number' ||
        !Number.isFinite(primitive.yaw_deg) ||
        Math.abs(primitive.yaw_deg) > 36000 ||
        (primitive.gripper !== undefined && !['open', 'closed', 'hold'].includes(primitive.gripper)) ||
        (primitive.force_limit_n !== undefined &&
          primitive.force_limit_n !== null &&
          (typeof primitive.force_limit_n !== 'number' ||
            !Number.isFinite(primitive.force_limit_n) ||
            primitive.force_limit_n < 0 ||
            primitive.force_limit_n > 10000))
      )
        return fail();
    }
  }
  for (const c of p.connections)
    if (
      !c ||
      typeof c !== 'object' ||
      !text(c.id) ||
      (!ids.has(c.lower) && c.lower !== 'baseplate') ||
      !ids.has(c.upper) ||
      !Array.isArray(c.studs) ||
      c.studs.length > 8192 ||
      !c.studs.every((v) => vec(v, 2))
    )
      return fail();
  for (const s of p.assembly.steps)
    if (!s || typeof s !== 'object' || !text(s.title) || !Array.isArray(s.brick_ids) || !s.brick_ids.every((id) => ids.has(id)))
      return fail();
  if (
    !p.validation ||
    typeof p.validation.valid !== 'boolean' ||
    !Array.isArray(p.validation.errors) ||
    !Array.isArray(p.validation.warnings) ||
    !p.metrics ||
    !p.execution ||
    !Array.isArray(p.execution.blockers)
  )
    return fail();
  if (
    p.metrics.bricks !== p.assembly.bricks.length ||
    p.metrics.steps !== p.steps.length ||
    p.metrics.connections !== p.connections.length ||
    p.metrics.primitives !== p.steps.reduce((n, s) => n + s.primitives.length, 0) ||
    !Number.isFinite(p.metrics.primitives) ||
    !Number.isFinite(p.metrics.stud_connections)
  )
    return fail();
  if (!p.validation.valid) throw new Error('This compiled plan has validation errors. Resolve them with the engine before previewing.');
  return p;
}
