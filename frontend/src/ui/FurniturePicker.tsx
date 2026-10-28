import { useEffect, useState } from 'react';
import { loadLibrary, type PlanSummary } from '../engine/plan';

export function FurniturePicker({ current, onPick }: { current: string; onPick: (id: string) => void }) {
  const [library, setLibrary] = useState<PlanSummary[]>([]);
  useEffect(() => {
    loadLibrary()
      .then(setLibrary)
      .catch(() => setLibrary([]));
  }, []);
  if (library.length < 2) return null;
  return (
    <label className="picker">
      <span className="visually-hidden">Furniture</span>
      <select value={current} onChange={(e) => onPick(e.target.value)}>
        {library.map((p) => (
          <option key={p.id} value={p.id}>
            {p.title}
          </option>
        ))}
      </select>
    </label>
  );
}
