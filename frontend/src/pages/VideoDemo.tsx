import { ArrowLeft, Pause, Play } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { loadPlan, type AssemblyPlan } from '../engine/plan';
import { loadSessionResult, loadSessionTruth, sessionVideoUrl, type SessionResult, type SessionTruth } from '../engine/results';
import { navigate } from '../navigation';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { BeliefBars } from '../ui/BeliefBars';
import { EvidenceHeatmap } from '../ui/EvidenceHeatmap';
import { FurniturePicker } from '../ui/FurniturePicker';

/**
 * The video analyzer, replayed: a rendered build session plays while what Muse Spark read on
 * each sampled frame (stored by `darius eval`) and the tracker's decoding of the whole video
 * follow it. The heatmap is the raw evidence (P(step done) per frame), the line through it the
 * decoded path, the ticks the ground truth; the graph fills in as the video reaches each joint.
 */

const ORIENTATION: Record<string, string> = {
  upright: 'upright',
  on_back: 'on its back',
  on_front: 'face down',
  on_side: 'on its side',
  upside_down: 'upside down',
  none: 'nothing joined yet',
};

function percent(value: number): string {
  return `${Math.round(value * 1000) / 10}`;
}

/** Ground truth used for scoring: a step is done once its parts are in place. */
function placedAt(truth: SessionTruth): Map<number, number> {
  return new Map(truth.segments.filter((s) => s.phase === 'place').map((s) => [s.step, s.start + s.duration]));
}

export function VideoDemo({ furniture }: { furniture: string }) {
  const [plan, setPlan] = useState<AssemblyPlan | null>(null);
  const [result, setResult] = useState<SessionResult | null | undefined>(undefined);
  const [truth, setTruth] = useState<SessionTruth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    Promise.all([loadPlan(furniture), loadSessionResult(furniture), loadSessionTruth(furniture)])
      .then(([p, r, t]) => {
        setPlan(p);
        setResult(r);
        setTruth(t);
      })
      .catch((e: Error) => setError(e.message));
  }, [furniture]);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      if (video.current) setTime(video.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const placed = useMemo(() => (truth ? placedAt(truth) : new Map<number, number>()), [truth]);

  if (error) return <p className="page-error">Couldn't load this session: {error}</p>;
  if (!plan || result === undefined) {
    return (
      <div className="screen-loading">
        <span className="loader" />
      </div>
    );
  }

  const steps = plan.steps.length;
  const duration = result?.duration_s ?? truth?.duration ?? 1;
  const index = result ? Math.min(result.frames.length - 1, Math.floor(time / result.every_s) - 1) : -1;
  const reading = result && index >= 0 ? result.frames[index] : null;
  const completed = result && index >= 0 ? result.path[index] : 0;
  const belief =
    result && index >= 0
      ? {
          probabilities: result.posteriors[index],
          completed,
          confidence: result.confidence[index],
        }
      : null;
  const truthDone = [...placed.values()].filter((t) => t <= time).length;
  const seek = (t: number) => {
    if (video.current) video.current.currentTime = Math.max(0, Math.min(duration - 0.05, t));
  };
  const toggle = () => {
    const v = video.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  return (
    <div className="demo video-demo">
      <header className="demo-top">
        <button type="button" className="icon-btn" onClick={() => navigate({ page: 'home' })} aria-label="Back">
          <ArrowLeft size={17} />
        </button>
        <div className="ar-title">
          <span className="eyebrow track">Video analyzer · replay of a real run</span>
          <strong>{plan.title}</strong>
        </div>
        <div className="demo-actions">
          {result && <span className="chip mono">{result.model}</span>}
          <FurniturePicker current={furniture} onPick={(id) => navigate({ page: 'video', furniture: id })} />
        </div>
      </header>

      <main className="video-grid">
        <section className="video-main">
          <div className="video-frame">
            <video
              ref={video}
              src={sessionVideoUrl(plan.id)}
              muted
              playsInline
              autoPlay
              loop
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onClick={toggle}
            />
            <div className="video-hud">
              <span className="chip">
                <span className="rec" /> <b>{time.toFixed(1)}</b> s
              </span>
              {result && (
                <span className={`chip ${completed === truthDone ? 'done' : 'guide'}`}>
                  <b>{`${completed}/${steps}`}</b> decoded · <b>{`${truthDone}/${steps}`}</b> true
                </span>
              )}
            </div>
            {reading && (
              <div className="video-reading glass">
                <span className="eyebrow track">
                  Frame {index + 1} · {reading.time_s.toFixed(0)} s
                </span>
                <div className="reading-letters">
                  <span className="faint mono">joined</span>
                  <span>
                    {reading.attached.length ? (
                      reading.attached.map((l, i) => (
                        <span key={i} className="letter is-on">
                          {l}
                        </span>
                      ))
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </span>
                  <span className="faint mono">loose</span>
                  <span>
                    {reading.loose.length ? (
                      reading.loose.map((l, i) => (
                        <span key={i} className="letter">
                          {l}
                        </span>
                      ))
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </span>
                  <span className="faint mono">lies</span>
                  <span>{ORIENTATION[reading.orientation] ?? reading.orientation}</span>
                </div>
              </div>
            )}
          </div>

          <div className="timeline glass panel-pad">
            <div className="panel-head">
              <span className="eyebrow">P(step done) per frame · decoded path · truth</span>
              <button type="button" className="icon-btn" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
                {playing ? <Pause size={15} /> : <Play size={15} />}
              </button>
            </div>
            {result ? (
              <EvidenceHeatmap plan={plan} result={result} placed={placed} time={time} onSeek={seek} />
            ) : (
              <p className="muted">
                No stored run. <code>darius eval --sessions --only {plan.id}</code> watches this video with the model.
              </p>
            )}
            <p className="panel-note">
              Cells: what the model said about each step on each sampled frame (brighter = done). Black line: the most likely build history
              (Viterbi). Diamonds: when the tracker says each step was finished; green ticks: when its parts were really in place.
            </p>
          </div>
        </section>

        <aside className="video-side">
          <div className="glass panel-pad">
            <div className="panel-head">
              <span className="eyebrow track">Assembly graph, as built</span>
              <span className="faint mono">
                {plan.steps.slice(0, completed).reduce((n, s) => n + s.joints.length, 0)}/{plan.joints.length} joints
              </span>
            </div>
            <AssemblyGraph plan={plan} completed={completed} active={completed < steps ? completed + 1 : null} />
          </div>
          {belief && (
            <div className="glass panel-pad">
              <div className="panel-head">
                <span className="eyebrow done">Tracker belief</span>
                <span className="faint mono">forward-backward · {Math.round(belief.confidence * 100)}%</span>
              </div>
              <BeliefBars plan={plan} belief={belief} />
            </div>
          )}
          {result && (
            <div className="glass panel-pad scorecard">
              <div className="panel-head">
                <span className="eyebrow done">Scored against the truth</span>
                <span className="faint mono">
                  {result.frames.length} frames · {result.seconds.toFixed(0)} s
                </span>
              </div>
              <div className="scores">
                <div>
                  <b>{percent(result.score.frame_accuracy)}</b>
                  <span>frames right</span>
                </div>
                <div>
                  <b>{percent(result.score.edges_recall)}</b>
                  <span>joints recovered</span>
                </div>
                <div>
                  <b className="plain">
                    {result.score.steps_found}/{result.score.steps}
                  </b>
                  <span>steps found</span>
                </div>
                <div>
                  <b className="plain">{result.score.step_error_s?.toFixed(2) ?? '—'} s</b>
                  <span>mean timing error</span>
                </div>
              </div>
              <p className="panel-note">
                One frame alone gets {percent(result.score.raw_frame_accuracy)}% right; the tracker decodes the whole video along the
                manual's order. A step counts as done once its parts are in place.
              </p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
