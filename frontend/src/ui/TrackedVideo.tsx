import { useEffect, useRef, useState } from 'react';
import type { TrackingResult } from '../engine/results';
import { MaskLayer } from './MaskLayer';

/**
 * A session video with SAM 3.1's masks over it, in step with playback. The layer redraws only
 * when the video reaches the next tracked frame.
 */
export function TrackedVideo({ src, tracks, labels = true }: { src: string; tracks: TrackingResult | null; labels?: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);

  useEffect(() => {
    if (!tracks) return;
    let raf = 0;
    let shown = -1;
    const tick = () => {
      const t = video.current?.currentTime ?? 0;
      const frame = Math.round(t * tracks.fps);
      if (frame !== shown) {
        shown = frame;
        setTime(t);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [tracks]);

  return (
    <div className="tracked-video">
      <video ref={video} src={src} muted loop autoPlay playsInline preload="metadata" />
      {tracks && <MaskLayer tracks={tracks} time={time} labels={labels} />}
    </div>
  );
}
