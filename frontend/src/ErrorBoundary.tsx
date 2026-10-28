import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * Catches a render error in one screen and shows a way back, instead of unmounting the whole app
 * to a blank page. Keyed by route in App, so moving to another screen starts fresh.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Darius: a screen failed to render', error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="screen-error" role="alert">
        <h1>Something went wrong on this page.</h1>
        <p className="muted">{this.state.error.message}</p>
        <div className="screen-error-actions">
          <button type="button" className="btn btn-dark" onClick={() => window.location.reload()}>
            Reload
          </button>
          <a className="text-link" href={import.meta.env.BASE_URL}>
            Back to the home page
          </a>
        </div>
      </div>
    );
  }
}
