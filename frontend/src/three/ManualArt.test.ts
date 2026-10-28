import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { plan } from '../test/library';
import { stepItems } from './ManualArt';

/** Where a part is pulled out to, in the drawing's world frame (meters). */
function pull(id: string, step: number, partId: string): THREE.Vector3 {
  const item = stepItems(plan(id), step).find((i) => i.part.id === partId);
  if (!item?.explode) throw new Error(`${partId} is not pulled out in step ${step}`);
  return item.explode.clone().normalize();
}

describe('manual drawings', () => {
  it('slides shelves between two sides in from the front', () => {
    // Bookcase step 5: the shelves go between both side panels, standing upright.
    expect(pull('halden-bookcase', 5, 'G1').z).toBeCloseTo(1);
  });

  it('pulls a part away from the part it joins when nothing is in the way', () => {
    // Bookcase step 1, on its side: top and bottom stand on the lying side panel, so they come down from above.
    expect(pull('halden-bookcase', 1, 'B').y).toBeCloseTo(1);
  });

  it('brings a drawer built on the side in from the front, as one piece', () => {
    const items = stepItems(plan('varde-nightstand'), 6).filter((i) => i.explode);
    expect(items.map((i) => i.part.label).sort()).toEqual(['E', 'F', 'F', 'G', 'H']);
    for (const item of items) expect(item.explode!.clone().normalize().z).toBeCloseTo(1);
    expect(new Set(items.map((i) => i.group))).toEqual(new Set(['side']));
  });

  it('draws only the subassembly a step works on', () => {
    // Nightstand step 4 builds the drawer box: the carcass is not in the picture.
    const labels = stepItems(plan('varde-nightstand'), 4).map((i) => i.part.label);
    expect(labels.sort()).toEqual(['F', 'F', 'G']);
  });
});
