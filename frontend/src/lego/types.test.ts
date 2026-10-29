import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  BUILD_IDS,
  brickFootprint,
  instructionUrl,
  isOfficialAssetUrl,
  parseCompiledPlan,
  phaseAt,
  renderPlacement,
  timelineAt,
  timelineDuration,
  type LegoPlan,
} from './types';

describe('LEGO playback boundaries', () => {
  it('clamps invalid and out-of-range time without a phantom seventh brick', () => {
    expect(timelineAt(-9, 6)).toEqual({ completed: 0, active: 0, fraction: 0, phase: 0 });
    expect(timelineAt(NaN, 6).completed).toBe(0);
    expect(timelineAt(31, 6)).toEqual({ completed: 6, active: 5, fraction: 1, phase: 5 });
    expect(timelineDuration(6)).toBe(30.6);
    expect(timelineAt(30.55, 6).completed).toBe(5);
    expect(timelineAt(30.6, 6).completed).toBe(6);
  });
  it('advances exactly one brick at every display cycle boundary', () => {
    expect(timelineAt(5.09, 6).completed).toBe(0);
    expect(timelineAt(5.1, 6)).toEqual({ completed: 1, active: 1, fraction: 0, phase: 0 });
    expect(timelineAt(12.5, 6).active).toBe(2);
    expect(phaseAt(0.9)).toBe(4);
  });
});

describe('official instruction discovery', () => {
  it('links directly to the official page for an ASCII set number', () => {
    expect(instructionUrl('10698')).toBe('https://www.lego.com/en-us/service/building-instructions/10698');
  });
  it('rejects URLs and input that could change the destination', () => {
    for (const value of ['https://other.example', '../12345', '123', '12345678', '10497?x=1'])
      expect(() => instructionUrl(value)).toThrow();
  });
});

describe('canonical compiler artifacts rendered by the frontend', () => {
  for (const id of BUILD_IDS)
    it(`${id} has complete instance and step geometry`, () => {
      const plan = JSON.parse(readFileSync(join(__dirname, '../../../assemblies/lego', id, 'plan.json'), 'utf8')) as LegoPlan;
      expect(parseCompiledPlan(plan)).toEqual(plan);
      const invalid = structuredClone(plan);
      invalid.assembly.bricks[0].studs = [10000, 10000];
      expect(() => parseCompiledPlan(invalid)).toThrow();
      expect(plan.validation.valid).toBe(true);
      expect(plan.steps.length).toBe(plan.assembly.bricks.length);
      for (const step of plan.steps) {
        const brick = plan.assembly.bricks.find((b) => b.id === step.brick_id);
        expect(brick).toBeDefined();
        expect(step.source_center_mm).toHaveLength(3);
        expect(step.target_center_mm).toHaveLength(3);
        const footprint = brickFootprint(brick!);
        expect(footprint).toEqual(brick!.rotation % 180 ? [brick!.studs[1], brick!.studs[0]] : brick!.studs);
        expect(step.dependencies.every((parent) => plan.steps.findIndex((s) => s.brick_id === parent) < plan.steps.indexOf(step))).toBe(
          true,
        );
      }
      expect(plan.metrics.bricks).toBe(plan.assembly.bricks.length);
    });
});

describe('bounded imported plans', () => {
  const fixture = () => JSON.parse(readFileSync(join(__dirname, '../../../assemblies/lego/signal-tower/plan.json'), 'utf8')) as LegoPlan;
  it('counts baseplate and brick studs before allocating scene meshes', () => {
    const plan = fixture();
    plan.assembly.baseplate.studs = [64, 64];
    plan.assembly.bricks.forEach((brick) => {
      brick.studs = [32, 32];
    });
    expect(() => parseCompiledPlan(plan)).toThrow('8,192 studs');
  });
  it.each([
    ['nonfinite position', { position_mm: [0, NaN, 0] }],
    ['nonfinite rotation', { yaw_deg: Infinity }],
    ['nonfinite duration', { duration_s: Infinity }],
    ['negative duration', { duration_s: -1 }],
    ['zero duration', { duration_s: 0 }],
    ['excessive duration', { duration_s: 3601 }],
    ['unsupported primitive', { kind: 'teleport' }],
    ['unsupported gripper state', { gripper: 'launch' }],
    ['nonfinite force', { force_limit_n: NaN }],
  ])('rejects %s', (_label, fields) => {
    const plan = fixture();
    Object.assign(plan.steps[0].primitives[0], fields);
    expect(() => parseCompiledPlan(plan)).toThrow('compiled Darius LEGO plan.json');
  });
  it('rejects a malformed target and invalid compiler output', () => {
    const malformed = fixture();
    malformed.steps[0].target_center_mm = [1, 2];
    expect(() => parseCompiledPlan(malformed)).toThrow();
    const invalid = fixture();
    invalid.validation.valid = false;
    expect(() => parseCompiledPlan(invalid)).toThrow('validation errors');
  });
});

describe('official instruction asset hosts', () => {
  it('accepts only the exact HTTPS hosts used by the server', () => {
    for (const url of [
      'https://www.lego.com/cdn/manual.pdf',
      'https://lego.com/file.pdf',
      'https://www.legocdn.com/file.pdf',
      'https://www.lego.com:443/file.pdf',
    ])
      expect(isOfficialAssetUrl(url)).toBe(true);
    for (const url of [
      'https://evil-lego.com/manual.pdf',
      'https://fakelego.com/file.pdf',
      'https://lego.com.evil.example/file.pdf',
      'https://user@www.lego.com/file.pdf',
      'https://www.lego.com:444/file.pdf',
      'http://www.lego.com/file.pdf',
      '//www.lego.com/file.pdf',
      'not a URL',
    ])
      expect(isOfficialAssetUrl(url)).toBe(false);
  });
});

describe('compiler-to-viewer coordinate contract', () => {
  it('honors explicit center heights, the 8mm scale, and relative source/target yaw', () => {
    const plan = JSON.parse(readFileSync(join(__dirname, '../../../assemblies/lego/signal-tower/plan.json'), 'utf8')) as LegoPlan;
    plan.assembly.baseplate.height_mm = 20;
    const step = plan.steps[0];
    step.target_center_mm = [56, 24.8, 48];
    step.source_center_mm = [168, 4.8, 40];
    step.source_yaw_deg = 0;
    step.target_yaw_deg = 90;
    const pose = renderPlacement(plan, step, plan.assembly.bricks[0]);
    expect(pose.target[0]).toBe(-1);
    expect(pose.target[1]).toBeCloseTo(2.5);
    expect(pose.target[2]).toBe(-2);
    expect(pose.source[0]).toBe(13);
    expect(pose.source[1]).toBeCloseTo(0);
    expect(pose.source[2]).toBe(-3);
    expect(pose.trayYaw).toBeCloseTo(-Math.PI / 2);
    delete step.target_center_mm;
    expect(renderPlacement(plan, step, plan.assembly.bricks[0]).target[1]).toBeCloseTo(2.5);
  });
});
