import type { AssemblyPlan } from '../engine/plan';
import type { Belief } from '../engine/tracking';

/** The tracker's posterior: one bar per "k steps done", the most probable one lit. */
export function BeliefBars({ plan, belief }: { plan: AssemblyPlan; belief: Belief }) {
  return (
    <ol className="belief" aria-label="Probability of each number of completed steps">
      {belief.probabilities.map((p, k) => (
        <li
          key={k}
          className={k === belief.completed ? 'is-map' : ''}
          title={k === 0 ? 'Nothing done' : `Done through step ${k}: ${plan.steps[k - 1].title}`}
        >
          <span className="belief-k">{k}</span>
          <span className="belief-track">
            <span className="belief-fill" style={{ width: `${Math.max(1.5, p * 100)}%` }} />
          </span>
          <span className="belief-p">{p < 0.001 ? '<.001' : p.toFixed(3).replace(/^0/, '')}</span>
        </li>
      ))}
    </ol>
  );
}
