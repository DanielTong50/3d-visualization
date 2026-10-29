/** Which screen to show. The URL holds it, so every screen can be reloaded and shared. */
export type Route =
  | { page: 'home' }
  | { page: 'legacy' }
  | { page: 'ar'; furniture: string }
  | { page: 'manual'; furniture: string }
  | { page: 'video'; furniture: string }
  | { page: 'live' }
  /** Bare views for scripts/capture.mjs: one manual page, or the clean session renderer. */
  | { page: 'render'; furniture: string; target: 'page' | 'session'; index: number };

export const DEFAULT_FURNITURE = 'halden-bookcase';
const SAFE_ID = /^[\w-]+$/;

export function routeFromUrl(): Route {
  const params = new URLSearchParams(window.location.search);
  const view = params.get('view');
  const furniture = params.get('furniture');
  const id = furniture && SAFE_ID.test(furniture) ? furniture : DEFAULT_FURNITURE;
  if (view === 'ar' || view === 'manual' || view === 'video') return { page: view, furniture: id };
  if (view === 'live') return { page: 'live' };
  if (view === 'legacy') return { page: 'legacy' };
  if (view === 'render') {
    const index = Number(params.get('page') ?? 0);
    return { page: 'render', furniture: id, target: index > 0 ? 'page' : 'session', index: Number.isInteger(index) ? index : 0 };
  }
  return { page: 'home' };
}

export function navigate(route: Route, { replace = false } = {}): void {
  const params = new URLSearchParams();
  if (route.page !== 'home') params.set('view', route.page);
  if ('furniture' in route) params.set('furniture', route.furniture);
  const query = params.toString();
  const url = `${window.location.pathname}${query ? `?${query}` : ''}`;
  if (replace) window.history.replaceState(null, '', url);
  else window.history.pushState(null, '', url);
  window.dispatchEvent(new PopStateEvent('popstate'));
  window.scrollTo?.(0, 0);
}
