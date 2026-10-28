import { Check, X } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { ApiError, DEFAULT_STAGES, getJob, type Job, type JobStage, type StageState } from './api';
import type { Upload } from './App';
import { Wordmark } from './Logo';
import { Pipeline } from './Pipeline';
import { jobSource, type StageName } from './replay';

const POLL_MS = 2000;

// What to try next when a step fails, keyed by the API's stage name.
const STAGE_ADVICE: Record<string, string> = {
  ingest: "We couldn't get enough sharp frames from the video. Film in brighter light and move the phone more slowly.",
  reconstruct:
    "We couldn't work out the 3D shape of the table. Walk a full circle around it and keep the whole table in view.",
  segment: "We couldn't find separate objects. Use 5–8 objects a few centimeters apart, and avoid glass and shiny metal.",
  lift: 'The objects were too close together to separate. Space them a few centimeters apart and film again.',
};
const GENERAL_ADVICE = 'Something went wrong on our side. Please try again in a few minutes.';

const STATE_WORDS: Record<StageState, string> = { pending: 'waiting', running: 'in progress', done: 'done', failed: 'failed' };

function StageRow({ label, state, detail }: { label: string; state: StageState; detail?: string | null }) {
  return (
    <li className={`stage stage-${state}`}>
      <span className="stage-icon" aria-hidden="true">
        {state === 'done' && <Check size={12} strokeWidth={3} />}
        {state === 'failed' && <X size={12} strokeWidth={3} />}
      </span>
      <span className="stage-label">
        {label}
        <span className="visually-hidden"> ({STATE_WORDS[state]})</span>
      </span>
      {detail && <span className="stage-time">{detail}</span>}
    </li>
  );
}

function StageList({ stages, uploadRow }: { stages: JobStage[]; uploadRow: ReactNode }) {
  return (
    <ol className="stages">
      {uploadRow}
      {stages.map((stage) => (
        <StageRow
          key={stage.name}
          label={stage.label}
          state={stage.state}
          detail={stage.seconds !== null && stage.state === 'done' ? `${Math.round(stage.seconds)} s` : null}
        />
      ))}
    </ol>
  );
}

/** How far along a scan is, from 0 to 1: the upload counts as one step, a running step as half done. */
function Progress({ value }: { value: number }) {
  const percent = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
      <div style={{ width: `${percent}%` }} />
    </div>
  );
}

function stagesProgress(stages: JobStage[]): number {
  const steps = stages.length + 1; // plus the finished upload
  const done = 1 + stages.reduce((sum, s) => sum + (s.state === 'done' ? 1 : s.state === 'running' ? 0.5 : 0), 0);
  return done / steps;
}

function Panel({
  eyebrow,
  title,
  onHome,
  children,
  below,
}: {
  eyebrow: string;
  title: string;
  onHome: () => void;
  children: ReactNode;
  /** Full-width content under the panel (the live pipeline). */
  below?: ReactNode;
}) {
  return (
    <main className="page site processing-page">
      <div className="processing" aria-live="polite">
        <div className="processing-brand">
          <Wordmark onClick={onHome} />
        </div>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
        </div>
        {children}
      </div>
      {below}
    </main>
  );
}

/** The job's stage graph with each stage's real output, filled in as the stages finish. */
function LivePipeline({ jobId, job }: { jobId: string; job: Job }) {
  const states = Object.fromEntries(job.stages.map((s) => [s.name, s.state])) as Partial<Record<StageName, StageState>>;
  const seconds = Object.fromEntries(job.stages.map((s) => [s.name, s.seconds])) as Partial<Record<StageName, number | null>>;
  // A new source (so views refetch) whenever a stage changes state.
  const version = job.stages.map((s) => `${s.name}:${s.state}`).join(',');
  const source = useMemo(() => jobSource(jobId, version), [jobId, version]);
  return (
    <section className="band band-dark processing-pipeline" aria-label="Live pipeline">
      <Pipeline source={source} states={states} seconds={seconds} />
    </section>
  );
}

function uploadProblem(error: ApiError): string {
  if (error.status === 0) return "The upload stopped: there's no connection to the server. Check your internet and try again.";
  if (error.status === 413) return 'That video is too large for the server. Try a shorter one (30–60 seconds).';
  if (error.status >= 500) return 'The server had a problem receiving the video. Please try again in a minute.';
  return `The server didn't accept the video: ${error.message}`;
}

/** Shown while a video uploads, before the server has a job for it. */
export function UploadScreen({ upload, onRetry, onHome }: { upload: Upload; onRetry: () => void; onHome: () => void }) {
  const percent = Math.round(upload.progress * 100);
  const failed = upload.error !== null;
  return (
    <Panel eyebrow="New scan" title={failed ? "The upload didn't finish" : 'Uploading your video'} onHome={onHome}>
      {failed ? (
        <p className="alert" role="alert">
          {uploadProblem(upload.error!)}
        </p>
      ) : (
        <>
          <Progress value={upload.progress / (DEFAULT_STAGES.length + 1)} />
          <p className="muted">Keep this page open. Big videos can take a minute on a phone connection.</p>
        </>
      )}
      <StageList
        stages={DEFAULT_STAGES}
        uploadRow={<StageRow label="Uploading your video" state={failed ? 'failed' : 'running'} detail={failed ? null : `${percent}%`} />}
      />
      <div className="actions">
        <button type="button" className={failed ? 'btn btn-ghost' : 'btn btn-secondary'} onClick={onHome}>
          {failed ? 'Back to home' : 'Cancel'}
        </button>
        {failed && (
          <button type="button" className="btn btn-primary" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    </Panel>
  );
}

function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

type JobView =
  | { kind: 'loading' }
  | { kind: 'job'; job: Job; reconnecting: boolean }
  | { kind: 'missing' };

/**
 * Follows a scan on the server: shows each step as it runs, explains failures, and swaps in the
 * playable scene when the job is done.
 */
export function JobScreen({ jobId, onHome, renderScene }: { jobId: string; onHome: () => void; renderScene: () => ReactNode }) {
  const [view, setView] = useState<JobView>({ kind: 'loading' });
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const job = await getJob(jobId);
        if (stopped) return;
        setView({ kind: 'job', job, reconnecting: false });
        if (job.state === 'done' || job.state === 'failed') return;
      } catch (error) {
        if (stopped) return;
        if (error instanceof ApiError && error.status === 404) {
          setView({ kind: 'missing' });
          return;
        }
        // Lost connection or a server hiccup: keep the last known progress and try again.
        setView((current) => (current.kind === 'job' ? { ...current, reconnecting: true } : current));
      }
      timer = window.setTimeout(poll, POLL_MS);
    };
    poll();
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      window.clearInterval(clock);
    };
  }, [jobId]);

  if (view.kind === 'loading') {
    return (
      <Panel eyebrow="Your scan" title="Checking your scan…" onHome={onHome}>
        <p className="muted">One moment.</p>
      </Panel>
    );
  }
  if (view.kind === 'missing') {
    return (
      <Panel eyebrow="Your scan" title="We couldn't find this scan" onHome={onHome}>
        <p className="muted">The link may be wrong, or the scan may have expired.</p>
        <div className="actions">
          <button type="button" className="btn btn-primary" onClick={onHome}>
            Back to home
          </button>
        </div>
      </Panel>
    );
  }

  const { job, reconnecting } = view;
  if (job.state === 'done') return renderScene();

  const uploadDone = <StageRow label="Uploading your video" state="done" />;
  if (job.state === 'failed') {
    const failedStage = job.stages.find((stage) => stage.state === 'failed') ?? job.stages.find((s) => s.name === job.failed_stage);
    const advice = (failedStage && STAGE_ADVICE[failedStage.name]) ?? GENERAL_ADVICE;
    return (
      <Panel eyebrow="Your scan" title="We couldn't finish this scan" onHome={onHome}>
        <p className="alert" role="alert">
          {failedStage ? `It stopped at "${failedStage.label}". ` : ''}
          {advice}
        </p>
        <StageList stages={job.stages} uploadRow={uploadDone} />
        <div className="actions">
          <button type="button" className="btn btn-primary" onClick={onHome}>
            Try another video
          </button>
        </div>
        {job.error && (
          <details className="details">
            <summary>Technical details</summary>
            <pre>{job.error.split('\n')[0].slice(0, 400)}</pre>
          </details>
        )}
      </Panel>
    );
  }

  const elapsed = formatElapsed(now / 1000 - job.created);
  return (
    <Panel eyebrow="Your scan" title="Building your world" onHome={onHome} below={<LivePipeline jobId={jobId} job={job} />}>
      <Progress value={stagesProgress(job.stages)} />
      <p className="muted">
        This usually takes about six minutes. Keep this page open: each stage's output appears below as it finishes, and your
        scene opens by itself when it's ready.
      </p>
      <StageList stages={job.stages} uploadRow={uploadDone} />
      <div className="processing-foot">
        <span className="mono faint small">
          {elapsed}
          {job.state === 'queued' ? ' · waiting for a free GPU' : ''}
          {reconnecting ? ' · reconnecting…' : ''}
        </span>
        <button type="button" className="btn btn-ghost" onClick={onHome}>
          Back to home
        </button>
      </div>
    </Panel>
  );
}
