import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { AssemblyPlan, PlanSummary } from '../engine/plan';
import type { ManualResult, SessionResult } from '../engine/results';

/** The repository's plan library, read from disk for tests. */
const LIBRARY = join(__dirname, '..', '..', '..', 'assemblies');

const read = <T>(...path: string[]): T => JSON.parse(readFileSync(join(LIBRARY, ...path), 'utf8')) as T;

export const library = (): PlanSummary[] => read<PlanSummary[]>('index.json');
export const plan = (id: string): AssemblyPlan => read<AssemblyPlan>(id, 'plan.json');
export const manualResult = (id: string): ManualResult => read<ManualResult>(id, 'manual', 'reading.json');
export const sessionResult = (id: string): SessionResult => read<SessionResult>(id, 'session', 'analysis.json');
