import type { ReactNode } from 'react';

/** Line-art pictograms of flat-pack hardware and tools, drawn like the ones in printed manuals. */

const SHAPES: Record<string, ReactNode> = {
  dowel: (
    <>
      <rect x="6" y="25" width="52" height="14" rx="6" />
      {[16, 24, 32, 40, 48].map((x) => (
        <path key={x} d={`M${x} 26 l-3 12`} />
      ))}
    </>
  ),
  cam_bolt: (
    <>
      <circle cx="13" cy="32" r="7" />
      <path d="M8 32 h10" />
      <rect x="20" y="28" width="12" height="8" />
      <path d="M32 29 h22 l3 3 l-3 3 h-22 z" />
      {[36, 41, 46, 51].map((x) => (
        <path key={x} d={`M${x} 29 l2 6`} />
      ))}
    </>
  ),
  cam_lock: (
    <>
      <circle cx="32" cy="32" r="20" />
      <circle cx="32" cy="32" r="13" />
      <path d="M23 32 h18 M37 28 l4 4 l-4 4" />
    </>
  ),
  nail: (
    <>
      <path d="M8 24 v16" strokeWidth="4" />
      <path d="M10 32 h40 l7 0" />
      <path d="M50 30.5 l7 1.5 l-7 1.5" />
    </>
  ),
  shelf_pin: (
    <>
      <rect x="12" y="24" width="6" height="16" rx="2" />
      <rect x="18" y="28" width="26" height="8" rx="4" />
    </>
  ),
  screw: (
    <>
      <path d="M6 22 v20 l10 -6 v-8 z" />
      <path d="M8 32 h6" />
      <path d="M16 29 h32 l8 3 l-8 3 h-32 z" />
      {[20, 25, 30, 35, 40, 45].map((x) => (
        <path key={x} d={`M${x} 29 l3 6`} />
      ))}
    </>
  ),
  bolt: (
    <>
      <path d="M6 24 h10 v16 h-10 z M6 29 h10 M6 35 h10" />
      <path d="M16 28 h40 v8 h-40 z" />
      {[24, 30, 36, 42, 48].map((x) => (
        <path key={x} d={`M${x} 28 l3 8`} />
      ))}
    </>
  ),
  screwdriver: (
    <>
      <rect x="4" y="25" width="22" height="14" rx="6" />
      <path d="M8 29 h14 M8 35 h14" />
      <path d="M26 30 h24 l6 2 l-6 2 h-24 z" />
    </>
  ),
  'allen key': (
    <>
      <path d="M14 12 v34 a6 6 0 0 0 6 6 h30" fill="none" strokeWidth="7" strokeLinecap="round" />
      <path d="M14 12 v34 a6 6 0 0 0 6 6 h30" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
    </>
  ),
  hammer: (
    <>
      <path d="M30 18 v38" strokeWidth="5" strokeLinecap="round" />
      <path d="M14 10 h30 l6 4 v8 h-36 z" />
    </>
  ),
};

export function HardwareIcon({ kind, size = 64 }: { kind: string; size?: number }) {
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} className="hw-icon" aria-hidden="true">
      <g fill="#fff" stroke="#141414" strokeWidth="2" strokeLinejoin="round">
        {SHAPES[kind] ?? <circle cx="32" cy="32" r="12" />}
      </g>
    </svg>
  );
}
