import { MATERIALS } from './materials';
import { BUMP_SPEEDS, GRAVITY_PRESETS, type BumpStrength, type GravityPreset } from './physics';
import type { Vec3 } from './scene';

/**
 * "What if" actions: Muse Spark answers a plain-English scenario with tool calls (see the API's
 * POST /what-if). They are model output, so every field is checked here before anything moves.
 */
export type WhatIfAction =
  | { name: 'apply_impulse'; objectId: string; direction: Vec3; speed: number }
  | { name: 'set_property'; objectId: string; property: 'mass_kg' | 'friction' | 'restitution'; value: number }
  | { name: 'set_material'; objectId: string; material: string }
  | { name: 'bump_table'; direction: Vec3; strength: BumpStrength }
  | { name: 'tilt_table'; degrees: number; axis: 'x' | 'z' }
  | { name: 'set_gravity'; preset: GravityPreset }
  | { name: 'drop_object'; objectId: string; height: number };

class Rejected extends Error {}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

function reject(reason: string): never {
  throw new Rejected(reason);
}

function number(value: unknown, field: string): number {
  const parsed = typeof value === 'string' ? Number(value) : value;
  if (typeof parsed !== 'number' || !Number.isFinite(parsed)) reject(`${field} is not a number`);
  return parsed;
}

function direction(value: unknown): Vec3 {
  if (!Array.isArray(value) || value.length !== 3) reject('direction is not [x, y, z]');
  const v = value.map((part, i) => number(part, `direction[${i}]`)) as Vec3;
  if (Math.hypot(...v) < 1e-6) reject('direction is zero');
  return v;
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], field: string): T {
  if (typeof value !== 'string' || !allowed.includes(value as T)) reject(`${field} must be one of ${allowed.join(', ')}`);
  return value as T;
}

const PROPERTY_RANGES = { mass_kg: [0.001, 200], friction: [0, 2], restitution: [0, 1] } as const;

function parseOne(raw: unknown, objectIds: ReadonlySet<string>): WhatIfAction {
  if (typeof raw !== 'object' || raw === null) reject('not an action');
  const { name } = raw as { name?: unknown };
  let args = (raw as { arguments?: unknown }).arguments ?? {};
  if (typeof args === 'string') {
    try {
      args = JSON.parse(args);
    } catch {
      reject('arguments are not JSON');
    }
  }
  if (typeof args !== 'object' || args === null) reject('arguments are not an object');
  const a = args as Record<string, unknown>;
  const objectId = () => {
    if (typeof a.object_id !== 'string' || !objectIds.has(a.object_id)) reject(`no object "${String(a.object_id)}"`);
    return a.object_id;
  };

  switch (name) {
    case 'apply_impulse':
      return { name, objectId: objectId(), direction: direction(a.direction), speed: clamp(number(a.speed_m_s, 'speed_m_s'), 0, 8) };
    case 'set_property': {
      const id = objectId();
      const property = oneOf(a.property, ['mass_kg', 'friction', 'restitution', 'material'] as const, 'property');
      if (property === 'material') {
        const material = typeof a.value === 'string' ? a.value.trim().toLowerCase() : '';
        if (!(material in MATERIALS)) reject(`unknown material "${String(a.value)}"`);
        return { name: 'set_material', objectId: id, material };
      }
      const [min, max] = PROPERTY_RANGES[property];
      return { name, objectId: id, property, value: clamp(number(a.value, 'value'), min, max) };
    }
    case 'bump_table':
      return { name, direction: direction(a.direction), strength: oneOf(a.strength, Object.keys(BUMP_SPEEDS) as BumpStrength[], 'strength') };
    case 'tilt_table':
      return { name, degrees: clamp(number(a.degrees, 'degrees'), -45, 45), axis: oneOf(a.axis, ['x', 'z'] as const, 'axis') };
    case 'set_gravity':
      return { name, preset: oneOf(a.preset, Object.keys(GRAVITY_PRESETS) as GravityPreset[], 'preset') };
    case 'drop_object':
      return { name, objectId: objectId(), height: clamp(number(a.height_m, 'height_m'), 0.05, 2) };
    default:
      reject(`unknown action "${String(name)}"`);
  }
}

/** Check a list of model-proposed actions: keep the valid ones, explain the rest. */
export function parseActions(raw: unknown, objectIds: ReadonlySet<string>): { actions: WhatIfAction[]; rejected: string[] } {
  const actions: WhatIfAction[] = [];
  const rejected: string[] = [];
  for (const item of Array.isArray(raw) ? raw : []) {
    try {
      actions.push(parseOne(item, objectIds));
    } catch (error) {
      if (!(error instanceof Rejected)) throw error;
      const name = typeof item === 'object' && item !== null ? String((item as { name?: unknown }).name) : '?';
      rejected.push(`${name}: ${error.message}`);
    }
  }
  return { actions, rejected };
}
