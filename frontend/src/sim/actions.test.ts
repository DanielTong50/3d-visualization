import { describe, expect, it } from 'vitest';
import { parseActions } from './actions';

const IDS = new Set(['obj_01', 'obj_02']);

describe('parseActions (model output is checked before anything moves)', () => {
  it('accepts every documented action', () => {
    const { actions, rejected } = parseActions(
      [
        { name: 'apply_impulse', arguments: { object_id: 'obj_01', direction: [1, 0, 0], speed_m_s: 2 } },
        { name: 'set_property', arguments: { object_id: 'obj_02', property: 'friction', value: '0.3' } },
        { name: 'set_property', arguments: { object_id: 'obj_02', property: 'material', value: ' Rubber ' } },
        { name: 'bump_table', arguments: { direction: [0, 0, 1], strength: 'hard' } },
        { name: 'tilt_table', arguments: { degrees: 20, axis: 'z' } },
        { name: 'set_gravity', arguments: { preset: 'moon' } },
        { name: 'drop_object', arguments: '{"object_id": "obj_01", "height_m": 0.4}' }, // arguments as a JSON string
      ],
      IDS,
    );
    expect(rejected).toEqual([]);
    expect(actions).toEqual([
      { name: 'apply_impulse', objectId: 'obj_01', direction: [1, 0, 0], speed: 2 },
      { name: 'set_property', objectId: 'obj_02', property: 'friction', value: 0.3 },
      { name: 'set_material', objectId: 'obj_02', material: 'rubber' },
      { name: 'bump_table', direction: [0, 0, 1], strength: 'hard' },
      { name: 'tilt_table', degrees: 20, axis: 'z' },
      { name: 'set_gravity', preset: 'moon' },
      { name: 'drop_object', objectId: 'obj_01', height: 0.4 },
    ]);
  });

  it('clamps numbers into their allowed ranges', () => {
    const { actions } = parseActions(
      [
        { name: 'apply_impulse', arguments: { object_id: 'obj_01', direction: [0, 1, 0], speed_m_s: 99 } },
        { name: 'tilt_table', arguments: { degrees: -80, axis: 'x' } },
        { name: 'set_property', arguments: { object_id: 'obj_01', property: 'restitution', value: 3 } },
      ],
      IDS,
    );
    expect(actions.map((a) => ('speed' in a ? a.speed : 'degrees' in a ? a.degrees : 'value' in a ? a.value : null))).toEqual([8, -45, 1]);
  });

  it('rejects unknown objects, actions, materials and malformed fields, with reasons', () => {
    const { actions, rejected } = parseActions(
      [
        { name: 'apply_impulse', arguments: { object_id: 'obj_99', direction: [1, 0, 0], speed_m_s: 1 } },
        { name: 'launch_rocket', arguments: {} },
        { name: 'set_property', arguments: { object_id: 'obj_01', property: 'material', value: 'unobtainium' } },
        { name: 'bump_table', arguments: { direction: [0, 0, 0], strength: 'light' } },
        { name: 'set_gravity', arguments: { preset: 'pluto' } },
        'not even an object',
      ],
      IDS,
    );
    expect(actions).toEqual([]);
    expect(rejected).toEqual([
      'apply_impulse: no object "obj_99"',
      'launch_rocket: unknown action "launch_rocket"',
      'set_property: unknown material "unobtainium"',
      'bump_table: direction is zero',
      'set_gravity: preset must be one of earth, moon, mars, jupiter, zero',
      '?: not an action',
    ]);
  });

  it('treats anything other than a list as no actions', () => {
    expect(parseActions({ name: 'set_gravity' }, IDS)).toEqual({ actions: [], rejected: [] });
  });
});
