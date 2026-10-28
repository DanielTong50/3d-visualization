import type { AssemblyPlan } from './plan';
import type { ManualResult as StoredManualResult } from './results';
import type { Observation } from './tracking';

/**
 * Client for the Darius API (engine/darius/service/api.py, deployed on Modal). Set VITE_API_URL
 * to use another server, e.g. a local one: uv run uvicorn darius.service.api:local_app --factory.
 */

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'https://danieltong12345--darius-api.us-east.modal.run';

export interface Health {
  ok: boolean;
  version: string;
  model: string;
  access_code_required: boolean;
  library: string[];
}

export interface FrameReading {
  frame: number;
  orientation: string;
  attached: string[];
  loose: string[];
  steps: { index: number; done: boolean; confidence: number }[];
}

/** What POST /manuals returns: the reading, the plan made from it, and what did not add up. */
export interface ManualReadResult {
  reading: StoredManualResult['reading'];
  plan: AssemblyPlan;
  notes: string[];
  model: string;
  pages: number;
  seconds: number;
}

const ACCESS_CODE_KEY = 'darius.accessCode';

export function savedAccessCode(): string {
  try {
    return localStorage.getItem(ACCESS_CODE_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveAccessCode(code: string): void {
  try {
    if (code) localStorage.setItem(ACCESS_CODE_KEY, code);
    else localStorage.removeItem(ACCESS_CODE_KEY);
  } catch {
    // Private mode: the code just isn't remembered.
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}, code = ''): Promise<T> {
  const headers = new Headers(init.headers);
  if (code) headers.set('X-Access-Code', code);
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError('The Darius API could not be reached.', 0);
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      // Not JSON: keep the status.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function health(): Promise<Health> {
  return call<Health>('/health');
}

/** Read photographed (or exported) manual pages into a plan. Takes about a minute. */
export function readManual(pages: Blob[], code: string): Promise<ManualReadResult> {
  const form = new FormData();
  pages.forEach((page, i) => form.append('pages', page, `page-${i + 1}.jpg`));
  return call<ManualReadResult>('/manuals', { method: 'POST', body: form }, code);
}

/** Check frames against a library plan (with its manual's step pages) or a plan read from photos. */
export async function checkFrames(
  frames: { image: Blob; timeS: number }[],
  plan: { id: string } | { plan: AssemblyPlan },
  code: string,
): Promise<{ readings: FrameReading[]; observations: Observation[] }> {
  const form = new FormData();
  frames.forEach((frame, i) => form.append('frames', frame.image, `frame-${i + 1}.jpg`));
  form.append('times', frames.map((f) => f.timeS.toFixed(2)).join(','));
  if ('id' in plan) form.append('plan_id', plan.id);
  else form.append('plan', JSON.stringify(plan.plan));
  const body = await call<{ readings: FrameReading[]; observations: { time_s: number; step_done: Record<string, number> }[] }>(
    '/frames',
    { method: 'POST', body: form },
    code,
  );
  return {
    readings: body.readings,
    observations: body.observations.map((o) => ({
      timeS: o.time_s,
      stepDone: Object.fromEntries(Object.entries(o.step_done).map(([k, p]) => [Number(k), p])),
    })),
  };
}

/** A frame of a video or camera element as a JPEG, at most `maxSide` pixels on its long side. */
export function grab(source: HTMLVideoElement, maxSide = 960): Promise<Blob | null> {
  const scale = Math.min(1, maxSide / Math.max(source.videoWidth, source.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(source.videoWidth * scale);
  canvas.height = Math.round(source.videoHeight * scale);
  if (!canvas.width || !canvas.height) return Promise.resolve(null);
  canvas.getContext('2d')!.drawImage(source, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.82));
}
