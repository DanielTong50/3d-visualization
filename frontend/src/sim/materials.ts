import table from '../../../pipeline/scan2sim/data/materials.json';
import type { SceneObject } from './scene';

/** Bulk properties of a material. The table is shared with the pipeline: pipeline/scan2sim/data/materials.json. */
export interface Material {
  density_kg_m3: number;
  /** Friction against a wooden table. */
  friction: number;
  /** Bounciness on a hard surface. */
  restitution: number;
  fragile: boolean;
  rigidity: 'rigid' | 'soft';
}

export const MATERIALS: Record<string, Material> = Object.fromEntries(
  Object.entries(table as Record<string, unknown>).filter(
    ([name, value]) => !name.startsWith('_') && typeof value === 'object' && value !== null,
  ),
) as Record<string, Material>;

/** The physical properties `spec` would have if it were made of `material`: same size and shape. */
export function materialChanges(spec: SceneObject, material: string): Partial<SceneObject> {
  const props = MATERIALS[material];
  if (!props) throw new Error(`unknown material "${material}"`);
  // Same volume and same hollowness, new density. Without a volume, scale the mass by density.
  const mass =
    spec.volume_m3 > 0
      ? spec.volume_m3 * props.density_kg_m3 * spec.fill_fraction
      : spec.density_kg_m3 > 0
        ? spec.mass_kg * (props.density_kg_m3 / spec.density_kg_m3)
        : spec.mass_kg;
  return {
    material,
    density_kg_m3: props.density_kg_m3,
    mass_kg: Math.min(Math.max(mass, 0.001), 200),
    friction: props.friction,
    restitution: props.restitution,
    fragile: props.fragile,
    rigidity: props.rigidity,
  };
}
