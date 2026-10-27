import { useEffect, useState, type ReactNode } from 'react';
import { ApiError, DEFAULT_STAGES, getJob, type Job, type JobStage, type StageState } from './api';
import type { Upload } from './App';

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
  const icon = { pending: '', running: '', done: '✓', failed: '✕' }[state];
  return (
    <li className={`stage stage-${state}`}>
      <span className="stage-icon" aria-hidden="true">
        {icon}
      </span>
      <span className="stage-label">
        {label}
        <span className="visually-hidden"> ({STATE_WORDS[state]})</span>
      </span>
      {detail && <span className="stage-detail">{detail}</span>}
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

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main className="page">
      <div className="processing" aria-live="polite">
        <h1>{title}</h1>
        {children}
      </div>
    </main>
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
    <Panel title={failed ? "The upload didn't finish" : 'Uploading your video'}>
      {failed ? (
        <p className="form-error" role="alert">
          {uploadProblem(upload.error!)}
        </p>
      ) : (
        <p className="muted">Keep this page open. Big videos can take a minute on a phone connection.</p>
      )}
      <StageList
        stages={DEFAULT_STAGES}
        uploadRow={<StageRow label="Uploading your video" state={failed ? 'failed' : 'running'} detail={failed ? null : `${percent}%`} />}
      />
      <div className="actions">
        {failed && (
          <button type="button" className="button" onClick={onRetry}>
            Try again
          </button>
        )}
        <button type="button" className={failed ? 'link-button' : 'button button-secondary'} onClick={onHome}>
          {failed ? 'Back to home' : 'Cancel'}
        </button>
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
      <Panel title="Checking your scan…">
        <p className="muted">One moment.</p>
      </Panel>
    );
  }
  if (view.kind === 'missing') {
    return (
      <Panel title="We couldn't find this scan">
        <p className="muted">The link may be wrong, or the scan may have expired.</p>
        <div className="actions">
          <button type="button" className="button" onClick={onHome}>
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
      <Panel title="We couldn't finish this scan">
        <p className="form-error" role="alert">
          {failedStage ? `It stopped at "${failedStage.label}". ` : ''}
          {advice}
        </p>
        <StageList stages={job.stages} uploadRow={uploadDone} />
        <div className="actions">
          <button type="button" className="button" onClick={onHome}>
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
    <Panel title="Building your world">
      <p className="muted">
        This usually takes 3–5 minutes. Keep this page open: your scene opens by itself when it's ready.
      </p>
      <StageList stages={job.stages} uploadRow={uploadDone} />
      <p className="muted small">
        Elapsed {elapsed}
        {job.state === 'queued' ? ' · waiting for a free computer' : ''}
        {reconnecting ? ' · reconnecting…' : ''}
      </p>
      <div className="actions">
        <button type="button" className="link-button" onClick={onHome}>
          Back to home
        </button>
      </div>
    </Panel>
  );
}
