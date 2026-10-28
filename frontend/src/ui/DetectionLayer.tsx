import { useEffect, useRef, useState } from 'react';
import type { Detection } from '../three/Stage';

/**
 * Detection boxes over the camera view, from the parts' projected 3D geometry: corner brackets
 * (cyan: tracked, orange: needed now, white: an assembly) with a label chip. Labels are placed
 * greedily so they never cover each other.
 */

const LABEL_H = 20;
const LABEL_GAP = 6;
const CHAR_W = 6.3;

interface Placed {
  x: number;
  y: number;
  w: number;
}

/** Each label sits above its box's top-left corner, pushed below any label it would cover. */
function placeLabels(detections: Detection[], width: number, height: number): Placed[] {
  const placed: Placed[] = [];
  for (const d of detections) {
    const w = 16 + (d.label ? 21 : 0) + (d.name.length + 5) * CHAR_W;
    const x = Math.min(Math.max(d.box.x * width, 6), width - w - 6);
    let y = Math.max(d.box.y * height, LABEL_H + LABEL_GAP + 34);
    for (let tries = 0; tries < detections.length; tries++) {
      const hit = placed.find((p) => x < p.x + p.w && x + w > p.x && Math.abs(y - p.y) < LABEL_H + 2);
      if (!hit) break;
      y = hit.y + LABEL_H + 2;
    }
    placed.push({ x, y, w });
  }
  return placed;
}

export function DetectionLayer({ detections }: { detections: Detection[] }) {
  const ref = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: 1440, height: 900 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setSize({ width: element.clientWidth, height: element.clientHeight }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const labels = placeLabels(detections, size.width, size.height);
  return (
    <svg ref={ref} className="detections" aria-hidden="true">
      {detections.map((d, i) => {
        const { x, y, w, h } = d.box;
        const arm = Math.min(0.018, w / 4, h / 4);
        const armY = (arm * size.width) / size.height;
        const corners = [
          `M${x},${y + armY} L${x},${y} L${x + arm},${y}`,
          `M${x + w - arm},${y} L${x + w},${y} L${x + w},${y + armY}`,
          `M${x + w},${y + h - armY} L${x + w},${y + h} L${x + w - arm},${y + h}`,
          `M${x + arm},${y + h} L${x},${y + h} L${x},${y + h - armY}`,
        ].join(' ');
        return (
          <g key={d.id} className={`det det-${d.role}`}>
            <svg x="0" y="0" width="100%" height="100%" viewBox="0 0 1 1" preserveAspectRatio="none" overflow="visible">
              <path d={corners} vectorEffect="non-scaling-stroke" />
              {d.role !== 'assembly' && <rect x={x} y={y} width={w} height={h} className="det-fill" />}
            </svg>
            <foreignObject x={labels[i].x} y={labels[i].y} width="1" height="1" overflow="visible">
              <span className="det-label">
                {d.label && <b>{d.label}</b>}
                {d.name}
                <i>{d.confidence.toFixed(2)}</i>
              </span>
            </foreignObject>
          </g>
        );
      })}
    </svg>
  );
}
