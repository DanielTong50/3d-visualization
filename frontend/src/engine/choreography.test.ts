import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { library, plan } from '../test/library';
import { Choreography } from './choreography';

describe.each(library().map((p) => p.id))('the choreography of %s', (id) => {
  const choreography = new Choreography(plan(id));
  const steps = choreography.plan.steps.length;

  it('runs its phases back to back, from the intro to the finished piece', () => {
    const { segments } = choreography;
    expect(segments[0].phase).toBe('intro');
    for (let i = 1; i < segments.length; i++) expect(segments[i].start).toBeCloseTo(segments[i - 1].start + segments[i - 1].duration);
    expect(choreography.moment(0).completed).toBe(0);
    const end = choreography.moment(choreography.duration);
    expect(end.phase).toBe('done');
    expect(end.completed).toBe(steps);
  });

  it('counts a step done only after it is verified', () => {
    for (const segment of choreography.segments.filter((s) => s.phase === 'verify')) {
      expect(choreography.moment(segment.start + segment.duration / 2).completed).toBe(segment.step - 1);
      expect(choreography.moment(segment.start + segment.duration + 0.01).completed).toBe(segment.step);
    }
  });

  it('ends with every part attached, rigidly in its place in the piece', () => {
    const states = choreography.partStates(choreography.moment(choreography.duration));
    const parts = choreography.plan.parts;
    expect([...states.values()].every((s) => s.status === 'attached')).toBe(true);
    // Relative placement matches the plan: the offset between two parts is the plan's offset.
    const at = (i: number) => new THREE.Vector3().setFromMatrixPosition(states.get(parts[i].id)!.matrix);
    const planned = (i: number) => new THREE.Vector3(...parts[i].assembled!.position).multiplyScalar(0.001);
    for (let i = 1; i < parts.length; i++) {
      expect(
        at(i)
          .sub(at(0))
          .distanceTo(planned(i).sub(planned(0))),
      ).toBeLessThan(1e-6);
    }
  });

  it('starts with every part lying loose on the floor', () => {
    const states = choreography.partStates(choreography.moment(0.1));
    expect([...states.values()].every((s) => s.status === 'loose')).toBe(true);
  });
});

describe.each(library().map((p) => p.id))('the hardware of %s', (id) => {
  const choreography = new Choreography(plan(id));
  const { pieces } = choreography;

  it('puts every piece of the inventory in the tray', () => {
    for (const item of choreography.plan.hardware) expect(pieces.filter((p) => p.hardware === item.id)).toHaveLength(item.count);
  });

  it('sends each piece in during its own step, while that step is fastened', () => {
    for (const piece of pieces.filter((p) => p.step !== null)) {
      const fasten = choreography.segments.find((s) => s.step === piece.step && s.phase === 'fasten')!;
      expect(piece.start).toBeGreaterThanOrEqual(fasten.start - 1e-9);
      expect(piece.end).toBeLessThanOrEqual(fasten.start + fasten.duration + 1e-9);
    }
  });

  it('ends with every used piece seated and every spare still in the tray', () => {
    const end = choreography.moment(choreography.duration);
    const poses = choreography.hardwarePoses(end, choreography.partStates(end));
    pieces.forEach((piece, i) => expect(poses[i].state).toBe(piece.step === null ? 'tray' : 'seated'));
  });

  it('keeps a seated piece with its part when the assembly turns', () => {
    const used = pieces.findIndex((p) => p.step === 1);
    if (used < 0) return;
    const turns = choreography.segments.filter((s) => s.phase === 'reorient');
    for (const turn of turns) {
      const before = choreography.moment(turn.start - 0.01);
      const after = choreography.moment(turn.start + turn.duration + 0.01);
      for (const m of [before, after]) {
        const states = choreography.partStates(m);
        const poses = choreography.hardwarePoses(m, states);
        if (poses[used].state !== 'seated') continue;
        const carrier = states.get(pieces[used].carrier!)!.matrix;
        const local = new THREE.Matrix4().copy(carrier).invert().multiply(poses[used].matrix);
        const expected = choreography.seated(pieces[used], states)!;
        expect(
          new THREE.Vector3().setFromMatrixPosition(poses[used].matrix).distanceTo(new THREE.Vector3().setFromMatrixPosition(expected)),
        ).toBeLessThan(1e-9);
        expect(Number.isFinite(local.elements[12])).toBe(true);
      }
    }
  });
});
