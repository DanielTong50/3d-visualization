import { X } from 'lucide-react';
import { useEffect } from 'react';
import { MATERIALS } from './sim/materials';
import { inertiaTensor, symmetricEigen } from './sim/robotics';
import type { SceneObject } from './sim/scene';

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatMass(kg: number): string {
  return kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`;
}

function frictionWord(friction: number): string {
  if (friction < 0.3) return 'Slippery';
  if (friction < 0.6) return 'Medium grip';
  return 'Grippy';
}

function bounceWord(restitution: number): string {
  if (restitution < 0.2) return 'Barely bounces';
  if (restitution < 0.5) return 'A little bouncy';
  return 'Very bouncy';
}

function formatSize([x, y, z]: [number, number, number]): string {
  return `${[x, z, y].map((m) => Math.round(m * 100)).join(' × ')} cm`;
}

/** Principal moments of inertia in kg·cm², smallest first (the eigenvalues of the tensor). */
function principalMoments(object: SceneObject): string | null {
  if (!object.inertial) return null;
  const { values } = symmetricEigen(inertiaTensor(object.inertial));
  return [...values]
    .sort((a, b) => a - b)
    .map((v) => (v * 1e4).toPrecision(2))
    .join(' · ');
}

/** How far the center of mass sits from the box center, in mm. */
function comOffset(object: SceneObject): string | null {
  if (!object.inertial) return null;
  const [x, y, z] = object.inertial.com_m;
  return `${Math.round(Math.hypot(x, y, z) * 1000)} mm off center`;
}

const SOURCE: Record<SceneObject['physics_source'], string> = {
  estimated: 'Estimated by AI from the scan',
  measured: 'Friction measured from a real push',
  'hand-authored': 'Values set by hand',
};

const MATERIAL_NAMES = Object.keys(MATERIALS).filter((name) => name !== 'unknown');

/** A short bar showing where a value sits between 0 and `max`. */
function Meter({ value, max, label }: { value: number; max: number; label: string }) {
  return (
    <span className="meter" role="meter" aria-valuemin={0} aria-valuemax={max} aria-valuenow={value} aria-label={label}>
      <span style={{ width: `${Math.min(value / max, 1) * 100}%` }} />
    </span>
  );
}

interface InspectorProps {
  object: SceneObject;
  /** True once the user (or a "what if") has changed this object's physics. */
  edited: boolean;
  onMaterial: (material: string) => void;
  onClose: () => void;
}

/** The panel for a tapped object: what it is, its physics, and a material to swap in. */
export function Inspector({ object, edited, onMaterial, onClose }: InspectorProps) {
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
    <aside className="inspector panel" aria-label={`${object.label} details`}>
      <div className="inspector-head">
        <div>
          <p className="eyebrow">Object</p>
          <h2>{capitalize(object.label)}</h2>
        </div>
        <button type="button" className="icon-btn plain" onClick={onClose} aria-label="Close">
          <X size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="inspector-tags">
        <span className="chip">{object.fragile ? 'Fragile' : 'Sturdy'}</span>
        {object.hollow && <span className="chip">Hollow</span>}
        {edited && <span className="chip chip-accent">Edited</span>}
      </div>
      <dl className="props">
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
        <dd title={frictionWord(object.friction)}>
          <Meter value={object.friction} max={1} label={`Friction: ${frictionWord(object.friction)}`} />
          {object.friction.toFixed(2)}
        </dd>
        <dt>Bounciness</dt>
        <dd title={bounceWord(object.restitution)}>
          <Meter value={object.restitution} max={1} label={`Bounciness: ${bounceWord(object.restitution)}`} />
          {object.restitution.toFixed(2)}
        </dd>
        <dt>Size</dt>
        <dd>{formatSize(object.dimensions_m)}</dd>
      </dl>
      {(object.inertial || object.grasps) && (
        <dl className="props props-robot" aria-label="For robots">
          {object.inertial && (
            <>
              <dt>Inertia</dt>
              <dd title="Principal moments of inertia, kg·cm², about the center of mass">
                {principalMoments(object)} <span className="unit">kg·cm²</span>
              </dd>
              <dt>Mass center</dt>
              <dd>
                {comOffset(object)}
                {object.inertial.model === 'shell' ? ' · hollow wall' : ''}
              </dd>
            </>
          )}
          {object.grasps && (
            <>
              <dt>Grasps</dt>
              <dd title="Parallel-jaw grasp candidates (140 mm stroke); press 4 for the robot view">
                {object.grasps.length === 0
                  ? 'Too wide for a parallel gripper'
                  : `${object.grasps.length} · best ${(object.grasps[0].width_m * 100).toFixed(1)} cm wide`}
              </dd>
            </>
          )}
        </dl>
      )}
      {object.physics_notes && !edited && <p className="inspector-note">{object.physics_notes}</p>}
      <p className="inspector-foot">
        {source}
        {edited ? '' : sureness}
      </p>
    </aside>
  );
}
