import { useMemo } from 'react';
import type { TrackFrame, TrackingResult } from '../engine/results';

/**
 * SAM 3.1's part tracks over the video: on the frame being shown, every tracked part's mask
 * outline, filled in its own color and tagged with the letter Muse Spark gave it. Around a
 * frame where Muse Spark prompted SAM (the anchor, or a later frame where it found a lost part
 * again), its box and point are drawn too. Coordinates are the tracking clip's pixels, so the
 * layer's viewBox is the clip size and it scales with the video.
 */

export const MASK_COLORS = [
  '#0058a3',
  '#d6361f',
  '#14a44d',
  '#e08a00',
  '#7b3fb5',
  '#00838f',
  '#c2185b',
  '#6d4c41',
  '#3949ab',
  '#689f38',
  '#f4511e',
  '#00897b',
  '#8e24aa',
  '#546e7a',
  '#1e88e5',
  '#9e9d24',
];

export const maskColor = (object: number) => MASK_COLORS[(object - 1) % MASK_COLORS.length];

/** How long (in frames, either side) a prompt stays drawn around its frame. */
const PROMPT_SPAN = 3;

function points(flat: number[]): string {
  let out = '';
  for (let i = 0; i + 1 < flat.length; i += 2) out += `${flat[i]},${flat[i + 1]} `;
  return out;
}

export function MaskLayer({ tracks, time, labels = true }: { tracks: TrackingResult; time: number; labels?: boolean }) {
  const byFrame = useMemo(() => tracks.tracks.map((t) => new Map<number, TrackFrame>(t.frames.map((f) => [f.frame, f]))), [tracks]);
  const [width, height] = tracks.size;
  const frame = Math.max(0, Math.min(tracks.frames - 1, Math.round(time * tracks.fps)));
  const prompts = tracks.prompts.filter((p) => Math.abs(p.frame - frame) <= PROMPT_SPAN);
  const tag = Math.round(width / 64);

  return (
    <svg className="masks" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {tracks.tracks.map((track, i) => {
        const record = byFrame[i].get(frame);
        if (!record || record.polygon.length < 6) return null;
        const color = maskColor(track.object);
        return <polygon key={track.object} className="mask" points={points(record.polygon)} style={{ fill: color, stroke: color }} />;
      })}
      {prompts.map((p) => {
        const [x0, y0, x1, y1] = p.box;
        const arm = Math.min(14, (x1 - x0) / 3, (y1 - y0) / 3);
        const d = [
          `M${x0},${y0 + arm}V${y0}H${x0 + arm}`,
          `M${x1 - arm},${y0}H${x1}V${y0 + arm}`,
          `M${x1},${y1 - arm}V${y1}H${x1 - arm}`,
          `M${x0 + arm},${y1}H${x0}V${y1 - arm}`,
        ].join('');
        return (
          <g key={`${p.object}-${p.frame}`} className="mask-prompt">
            <path d={d} />
            {p.point && <circle cx={p.point[0]} cy={p.point[1]} r={4} />}
          </g>
        );
      })}
      {labels &&
        tracks.tracks.map((track, i) => {
          const record = byFrame[i].get(frame);
          if (!record || record.polygon.length < 6) return null;
          const x = Math.min(Math.max(record.box[0], 2), width - tag * 2.6);
          const y = Math.min(Math.max(record.box[1] - tag - 4, 2), height - tag - 2);
          return (
            <g key={track.object} className="mask-tag" transform={`translate(${x} ${y})`}>
              <rect width={tag * 2.5} height={tag} rx={2} />
              <rect width={tag} height={tag} rx={2} className="mask-letter" style={{ fill: maskColor(track.object) }} />
              <text x={tag / 2} y={tag * 0.72} className="mask-letter-text" fontSize={tag * 0.66}>
                {track.label}
              </text>
              <text x={tag * 1.75} y={tag * 0.72} fontSize={tag * 0.6}>
                {track.object}
              </text>
            </g>
          );
        })}
    </svg>
  );
}
