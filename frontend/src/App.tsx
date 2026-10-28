import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { ApiError, createJob, jobSceneDir, saveAccessCode } from './api';
import { Home } from './Home';
import { navigate, routeFromUrl, SCENES_URL, type Route } from './navigation';
import { JobScreen, UploadScreen } from './Processing';

// The 3D engine (three.js, Rapier, Spark) is most of the app's weight: load it only for a scene,
// so the home and processing screens appear at once.
const SimulatorView = lazy(() => import('./SimulatorView').then((module) => ({ default: module.SimulatorView })));

function Scene({ sceneDir, view }: { sceneDir: string; view?: string }) {
  return (
    <Suspense
      fallback={
        <div className="sim">
          <div className="center-message" role="status">
            <span className="loader" aria-hidden="true" />
            <span className="muted">Loading scene…</span>
          </div>
        </div>
      }
    >
      <SimulatorView sceneDir={sceneDir} onHome={goHome} initialView={view} />
    </Suspense>
  );
}

export interface Upload {
  file: File;
  accessCode: string | null;
  progress: number;
  error: ApiError | null;
}

const goHome = () => navigate({ page: 'home' });

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromUrl);
  const [upload, setUpload] = useState<Upload | null>(null);
  const cancelUpload = useRef<(() => void) | null>(null);
  const [needsCode, setNeedsCode] = useState<{ file: File; wrongCode: boolean } | null>(null);

  useEffect(() => {
    const onPopState = () => setRoute(routeFromUrl());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const startUpload = (file: File, accessCode: string | null) => {
    setNeedsCode(null);
    setUpload({ file, accessCode, progress: 0, error: null });
    const title = `My scan, ${new Date().toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}`;
    const job = createJob(file, title, accessCode, (progress) => setUpload((u) => u && { ...u, progress }));
    cancelUpload.current = job.cancel;
    job.done
      .then((jobId) => {
        if (accessCode) saveAccessCode(accessCode);
        setUpload(null);
        navigate({ page: 'job', jobId }, { replace: true });
      })
      .catch((error: ApiError) => {
        if (error.status === -1) return; // cancelled
        if (error.status === 401 || error.status === 403) {
          saveAccessCode(null);
          setUpload(null);
          setNeedsCode({ file, wrongCode: accessCode !== null });
          navigate({ page: 'home' }, { replace: true });
          return;
        }
        setUpload((u) => u && { ...u, error });
      });
  };

  const leaveUpload = () => {
    cancelUpload.current?.();
    setUpload(null);
    goHome();
  };

  if (upload) {
    return <UploadScreen upload={upload} onRetry={() => startUpload(upload.file, upload.accessCode)} onHome={leaveUpload} />;
  }
  switch (route.page) {
    case 'scene':
      return <Scene key={route.sceneId} sceneDir={`${SCENES_URL}/${route.sceneId}`} view={route.view} />;
    case 'job':
      return (
        <JobScreen
          key={route.jobId}
          jobId={route.jobId}
          onHome={goHome}
          renderScene={() => <Scene sceneDir={jobSceneDir(route.jobId)} />}
        />
      );
    case 'home':
      return <Home onStart={startUpload} needsCode={needsCode} onCancelCode={() => setNeedsCode(null)} />;
  }
}
