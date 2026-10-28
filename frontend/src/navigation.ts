/** Which screen to show. The URL holds it, so every screen can be reloaded and shared. */
export type Route = { page: 'home' } | { page: 'scene'; sceneId: string; view?: string } | { page: 'job'; jobId: string };

const SAFE_ID = /^[\w-]+$/;

export function routeFromUrl(): Route {
  const params = new URLSearchParams(window.location.search);
  const job = params.get('job');
  const scene = params.get('scene');
  if (job && SAFE_ID.test(job)) return { page: 'job', jobId: job };
  const view = params.get('view');
  if (scene && SAFE_ID.test(scene)) return { page: 'scene', sceneId: scene, ...(view && SAFE_ID.test(view) ? { view } : {}) };
  return { page: 'home' };
}

/** Go to a screen. `replace` swaps the current history entry (e.g. upload → its job). */
export function navigate(route: Route, { replace = false } = {}): void {
  const params = new URLSearchParams();
  if (route.page === 'scene') params.set('scene', route.sceneId);
  if (route.page === 'scene' && route.view) params.set('view', route.view);
  if (route.page === 'job') params.set('job', route.jobId);
  if (new URLSearchParams(window.location.search).has('debug')) params.set('debug', '1');
  const query = params.toString();
  const url = `${window.location.pathname}${query ? `?${query}` : ''}`;
  if (replace) window.history.replaceState(null, '', url);
  else window.history.pushState(null, '', url);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

/** Scene packages are served next to the app, so this works under any base path (e.g. GitHub Pages). */
export const SCENES_URL = `${import.meta.env.BASE_URL}scenes`;
