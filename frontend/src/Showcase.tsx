import { useEffect, useRef } from 'react';
import { Simulator } from './sim/Simulator';

const ignore = () => {};
const CROSSFADE_MS = 700;

/**
 * A hands-off, slowly swaying view of a scene for the home page. Changing `sceneDir` loads the
 * next scene behind the current one and crossfades once it is ready, so the hero never goes blank.
 * It stops drawing while scrolled out of view, to spare phone batteries.
 */
export function Showcase({ sceneDir, onReady }: { sceneDir: string; onReady?: (sceneDir: string) => void }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const live = useRef<{ simulator: Simulator; layer: HTMLDivElement } | null>(null);
  const visible = useRef(true);
  const readyCallback = useRef(onReady);
  readyCallback.current = onReady;

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    let cancelled = false;
    const layer = document.createElement('div');
    layer.className = 'showcase-layer';
    root.appendChild(layer);
    Simulator.create(layer, sceneDir, { onSelect: ignore, onFps: ignore }, { showcase: true })
      .then((simulator) => {
        if (cancelled) {
          simulator.dispose();
          layer.remove();
          return;
        }
        const previous = live.current;
        live.current = { simulator, layer };
        simulator.setPaused(!visible.current);
        requestAnimationFrame(() => layer.classList.add('is-ready'));
        if (previous) {
          window.setTimeout(() => {
            previous.simulator.dispose();
            previous.layer.remove();
          }, CROSSFADE_MS);
        }
        readyCallback.current?.(sceneDir);
      })
      .catch((error: unknown) => {
        layer.remove();
        console.warn('Showcase scene failed to load:', error); // the page reads fine without it
      });
    return () => {
      cancelled = true;
    };
  }, [sceneDir]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const observer = new IntersectionObserver(([entry]) => {
      visible.current = entry.isIntersecting;
      live.current?.simulator.setPaused(!entry.isIntersecting);
    });
    observer.observe(root);
    return () => {
      observer.disconnect();
      live.current?.simulator.dispose();
      live.current?.layer.remove();
      live.current = null;
    };
  }, []);

  return <div ref={rootRef} className="viewport showcase" aria-hidden="true" />;
}
