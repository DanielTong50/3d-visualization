import { ArrowLeft, Pause, Play } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { loadPlan, type AssemblyPlan } from '../engine/plan';
import {
  contactGraph,
  loadSessionResult,
  loadSessionTruth,
  loadTrackingResult,
  sessionVideoUrl,
  type SessionResult,
  type SessionTruth,
  type TrackingResult,
} from '../engine/results';
import { navigate } from '../navigation';
import { AssemblyGraph } from '../ui/AssemblyGraph';
import { BeliefBars } from '../ui/BeliefBars';
import { EvidenceHeatmap } from '../ui/EvidenceHeatmap';
import { FurniturePicker } from '../ui/FurniturePicker';
import { MaskLayer } from '../ui/MaskLayer';

/**
 * The video analyzer, replayed: a rendered build session plays while what the models made of it
 * (stored by `darius eval`) follows it. Over the video, SAM 3.1's mask for every part, tracked
 * from Muse Spark's prompts. Below, what Muse Spark read on each sampled frame: the heatmap is
 * the raw evidence (P(step done) per frame), the line through it the decoded path, the ticks the
 * ground truth. At the side, the graph as built, and the graph SAM 3.1's contacts show.
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
  const [tracks, setTracks] = useState<TrackingResult | null>(null);
  const [masks, setMasks] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setTracks(null);
    Promise.all([loadPlan(furniture), loadSessionResult(furniture), loadSessionTruth(furniture)])
      .then(([p, r, t]) => {
        setPlan(p);
        setResult(r);
        setTruth(t);
      })
      .catch((e: Error) => setError(e.message));
    // The tracks are large and optional: they arrive after the page is up.
    loadTrackingResult(furniture)
      .then((t) => t?.plan === furniture && setTracks(t))
      .catch(() => setTracks(null));
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
  const contacts = tracks && tracks.plan === plan.id ? contactGraph(plan, tracks, time) : null;
  const refound = tracks ? tracks.prompts.filter((p) => p.frame !== tracks.anchor_frame).length : 0;
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
          {tracks && <span className="chip mono">{tracks.model}</span>}
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
            {tracks && masks && tracks.plan === plan.id && <MaskLayer tracks={tracks} time={time} />}
            <div className="video-hud">
              <span className="chip">
                <span className="rec" /> <b>{time.toFixed(1)}</b> s
              </span>
              {result && (
                <span className={`chip ${completed === truthDone ? 'done' : 'guide'}`}>
                  <b>{`${completed}/${steps}`}</b> decoded · <b>{`${truthDone}/${steps}`}</b> true
                </span>
              )}
              {tracks && (
                <button type="button" className={`chip${masks ? '' : ' is-off'}`} onClick={() => setMasks((m) => !m)} aria-pressed={masks}>
                  <b>{tracks.tracks.length}</b> parts · masks {masks ? 'on' : 'off'}
                </button>
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
          {tracks && contacts && (
            <div className="glass panel-pad">
              <div className="panel-head">
                <span className="eyebrow track">Graph from SAM 3.1 contacts</span>
                <span className="faint mono">
                  {contacts.seen.size}/{plan.joints.length} joints
                </span>
              </div>
              <AssemblyGraph plan={plan} completed={steps} known={contacts.seen} extra={contacts.extra} />
              <p className="panel-note">
                An edge appears once two parts' masks have touched for a second without a break. Red: parts touching in the image that no
                joint joins. Tracks are named by the part they cover on the first frame.
              </p>
            </div>
          )}
          {tracks && (
            <div className="glass panel-pad scorecard">
              <div className="panel-head">
                <span className="eyebrow done">Part tracking, scored</span>
                <span className="faint mono">
                  {tracks.frames} frames · {tracks.seconds.toFixed(0)} s
                </span>
              </div>
              <div className="scores">
                <div>
                  <b>{percent(tracks.score.mean_iou)}</b>
                  <span>mask IoU</span>
                </div>
                <div>
                  <b>{percent(tracks.score.frames_tracked)}</b>
                  <span>part-frames held</span>
                </div>
                <div>
                  <b className="plain">
                    {tracks.score.letters_right}/{tracks.score.pieces}
                  </b>
                  <span>letters right*</span>
                </div>
                <div>
                  <b>{percent(tracks.score.contacts_recall)}</b>
                  <span>joints touched</span>
                </div>
              </div>
              <div className="contact-runs">
                <span>Muse Spark readings alone</span>
                <b>{percent(tracks.score.frames_right_spark)}%</b>
                <span>SAM 3.1 contacts alone</span>
                <b>{percent(tracks.score.frames_right_contacts)}%</b>
                <span>Both, fused</span>
                <b>{percent(tracks.score.frames_right_fused)}%</b>
                <span className="is-bound">Contacts of exact masks</span>
                <b className="is-bound">{percent(tracks.score.frames_right_exact_contacts)}%</b>
              </div>
              <p className="panel-note">
                Muse Spark grounded {tracks.score.grounded} parts on the first frame
                {refound > 0 ? ` and found lost ones again ${refound} times later` : ''}; SAM 3.1 tracked them. Masks are scored against the
                renderer's exact part masks wherever a part is visible; frames right compare the timeline decoded from each source of
                evidence. *Up to look-alikes: parts of the same size, which only where they go tells apart.
              </p>
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
