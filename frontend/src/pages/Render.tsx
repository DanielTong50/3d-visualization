import { useEffect, useRef, useState } from 'react';
import { loadPlan, type AssemblyPlan } from '../engine/plan';
import { Stage } from '../three/Stage';
import { ManualPage, manualPages } from '../ui/ManualPage';

/**
 * Bare views that scripts/capture.mjs photographs: a manual page at print size, or the clean
 * session renderer (no AR layer, no HUD) driven frame by frame. The session view exposes
 * `window.__darius`, so the script can step the clock, read the ground truth of every frame, and
 * render the parts' exact label maps for scoring part trackers.
 */

declare global {
  interface Window {
    __darius?: {
      duration: number;
      segments: { step: number; phase: string; start: number; duration: number }[];
      /** Lay out and render time t; returns how many steps are done at t. */
      draw: (t: number) => { completed: number; step: number; phase: string };
      /** The parts' label map at time t (see Stage.renderIds), base64, and the part ids in label order. */
      ids: (t: number, width: number, height: number) => { labels: string; parts: string[] };
    };
  }
}

export function Render({ furniture, target, index }: { furniture: string; target: 'page' | 'session'; index: number }) {
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.classList.add('render');
    loadPlan(furniture).then(setPlan);
    return () => document.documentElement.classList.remove('render');
  }, [furniture]);

  useEffect(() => {
    if (!plan) return;
    if (target === 'page') {
      document.documentElement.dataset.ready = 'true';
      return;
    }
    const stage = new Stage(viewport.current!, plan, 'clean');
    window.__darius = {
      duration: stage.choreography.duration,
      segments: stage.choreography.segments.map((s) => ({ ...s })),
      draw: (t) => {
        const { moment } = stage.update(t, t);
        stage.render();
        return { completed: moment.completed, step: moment.step, phase: moment.phase };
      },
      ids: (t, width, height) => {
        const labels = stage.renderIds(t, width, height);
        let binary = '';
        for (let i = 0; i < labels.length; i += 0x8000) binary += String.fromCharCode(...labels.subarray(i, i + 0x8000));
        return { labels: btoa(binary), parts: plan.parts.map((p) => p.id) };
      },
    };
    document.documentElement.dataset.ready = 'true';
    return () => {
      delete window.__darius;
      stage.dispose();
    };
  }, [plan, target]);

  if (!plan) return null;
  if (target === 'page') {
    const spec = manualPages(plan)[index - 1];
    return spec ? <ManualPage plan={plan} spec={spec} /> : <p>No page {index}</p>;
  }
  return <div ref={viewport} className="render-session" />;
}
