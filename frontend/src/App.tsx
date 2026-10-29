import { lazy, Suspense, useEffect, useState } from 'react';
import { ErrorBoundary } from './ErrorBoundary';
import { routeFromUrl, type Route } from './navigation';

// Each screen loads on demand: the 3D stage (three.js) is most of the app's weight.
const Home = lazy(() => import('./pages/LegoLab').then((m) => ({ default: m.LegoLab })));
const LegacyHome = lazy(() => import('./pages/Home').then((m) => ({ default: m.Home })));
const ArDemo = lazy(() => import('./pages/ArDemo').then((m) => ({ default: m.ArDemo })));
const ManualDemo = lazy(() => import('./pages/ManualDemo').then((m) => ({ default: m.ManualDemo })));
const VideoDemo = lazy(() => import('./pages/VideoDemo').then((m) => ({ default: m.VideoDemo })));
const Live = lazy(() => import('./pages/Live').then((m) => ({ default: m.Live })));
const Render = lazy(() => import('./pages/Render').then((m) => ({ default: m.Render })));

function Loading() {
  return (
    <div className="screen-loading" role="status">
      <span className="loader" aria-hidden="true" />
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromUrl);
  useEffect(() => {
    const onPopState = () => setRoute(routeFromUrl());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);
  return (
    <ErrorBoundary key={JSON.stringify(route)}>
      <Suspense fallback={<Loading />}>
        {route.page === 'home' && <Home />}
        {route.page === 'legacy' && <LegacyHome />}
        {route.page === 'ar' && <ArDemo key={route.furniture} furniture={route.furniture} />}
        {route.page === 'manual' && <ManualDemo key={route.furniture} furniture={route.furniture} />}
        {route.page === 'video' && <VideoDemo key={route.furniture} furniture={route.furniture} />}
        {route.page === 'live' && <Live />}
        {route.page === 'render' && <Render furniture={route.furniture} target={route.target} index={route.index} />}
      </Suspense>
    </ErrorBoundary>
  );
}
