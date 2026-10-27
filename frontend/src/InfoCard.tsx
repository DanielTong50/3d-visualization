import { useEffect } from 'react';
import { MATERIALS } from './sim/materials';
import type { SceneObject } from './sim/scene';

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatMass(kg: number): string {
  return kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`;
}

function frictionWord(friction: number): string {
  if (friction < 0.3) return 'slippery';
  if (friction < 0.6) return 'medium grip';
  return 'grippy';
}

function bounceWord(restitution: number): string {
  if (restitution < 0.2) return 'barely bounces';
  if (restitution < 0.5) return 'a little bouncy';
  return 'very bouncy';
}

function formatSize([x, y, z]: [number, number, number]): string {
  return `${[x, z, y].map((m) => Math.round(m * 100)).join(' × ')} cm`;
}

const SOURCE: Record<SceneObject['physics_source'], string> = {
  estimated: 'Estimated by AI from the scan',
  measured: 'Friction measured from a real push',
  'hand-authored': 'Values set by hand',
};

const MATERIAL_NAMES = Object.keys(MATERIALS).filter((name) => name !== 'unknown');

interface InfoCardProps {
  object: SceneObject;
  /** True once the user (or a "what if") has changed this object's physics. */
  edited: boolean;
  onMaterial: (material: string) => void;
  onClose: () => void;
}

/** The details card for a tapped object: what it is, its physics, and a material to swap in. */
export function InfoCard({ object, edited, onMaterial, onClose }: InfoCardProps) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  const source = edited ? 'Changed by you' : (SOURCE[object.physics_source] ?? 'Estimated');
  const sureness =
    object.physics_source === 'hand-authored' ? '' : ` · ${Math.round(object.confidence * 100)}% sure it's a ${object.label}`;
  const materials = MATERIAL_NAMES.includes(object.material) ? MATERIAL_NAMES : [object.material, ...MATERIAL_NAMES];
  return (
    <aside className="info-card" aria-label={`${object.label} details`}>
      <div className="info-card-head">
        <h2>{capitalize(object.label)}</h2>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <p className="info-card-sub">{object.fragile ? 'Fragile' : 'Sturdy'}</p>
      <dl>
        <dt>
          <label htmlFor="material">Material</label>
        </dt>
        <dd>
          <select id="material" className="select" value={object.material} onChange={(event) => onMaterial(event.target.value)}>
            {materials.map((name) => (
              <option key={name} value={name}>
                {capitalize(name)}
              </option>
            ))}
          </select>
        </dd>
        <dt>Mass</dt>
        <dd>{formatMass(object.mass_kg)}</dd>
        <dt>Friction</dt>
        <dd>
          {object.friction.toFixed(2)} <span className="muted">{frictionWord(object.friction)}</span>
        </dd>
        <dt>Bounciness</dt>
        <dd>
          {object.restitution.toFixed(2)} <span className="muted">{bounceWord(object.restitution)}</span>
        </dd>
        <dt>Size</dt>
        <dd>{formatSize(object.dimensions_m)}</dd>
      </dl>
      {object.physics_notes && !edited && <p className="info-card-note">{object.physics_notes}</p>}
      <p className="info-card-foot">
        {source}
        {edited ? '' : sureness}
      </p>
    </aside>
  );
}
