/** The Darius mark (a solid block casting its outline: the real object and its simulated twin) and name. */
export function Wordmark({ href, onClick }: { href?: string; onClick?: () => void }) {
  const mark = (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <rect x="0.75" y="0.75" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <rect x="6" y="6" width="12" height="12" fill="currentColor" />
    </svg>
  );
  if (onClick) {
    return (
      <button type="button" className="wordmark plain-button" onClick={onClick}>
        {mark}
        darius
      </button>
    );
  }
  return (
    <a className="wordmark" href={href}>
      {mark}
      darius
    </a>
  );
}
