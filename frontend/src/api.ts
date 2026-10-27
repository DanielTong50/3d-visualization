/**
 * The processing API (FastAPI on Modal, owned by the pipeline). Set VITE_API_URL to use another
 * deployment, e.g. a local one. Endpoints: GET /health, POST /jobs, GET /jobs/{id},
 * GET /jobs/{id}/scene/{file}.
 */
export const API_URL: string = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'https://danieltong12345--scan2sim-api.us-east.modal.run'
).replace(/\/+$/, '');

export type StageState = 'pending' | 'running' | 'done' | 'failed';

export interface JobStage {
  name: string;
  /** Plain-language name of the step, shown on the processing screen. */
  label: string;
  state: StageState;
  seconds: number | null;
}

export interface Job {
  job_id: string;
  state: 'queued' | 'running' | 'done' | 'failed';
  /** Unix time in seconds. */
  created: number;
  stages: JobStage[];
  error: string | null;
  failed_stage?: string | null;
}

/** The steps a scan goes through, for showing before the API has reported any (same labels as the API). */
export const DEFAULT_STAGES: JobStage[] = [
  'Picking sharp frames',
  'Finding camera positions and building 3D',
  'Finding the objects',
  'Separating objects in 3D',
  'Identifying objects and their physics',
  'Building your world',
].map((label) => ({ name: label, label, state: 'pending', seconds: null }));

export class ApiError extends Error {
  constructor(
    message: string,
    /** HTTP status; 0 means the request never got an answer. */
    readonly status: number,
  ) {
    super(message);
  }
}

/** Where a finished job's scene package can be loaded from. */
export function jobSceneDir(jobId: string): string {
  return `${API_URL}/jobs/${encodeURIComponent(jobId)}/scene`;
}

export async function getJob(jobId: string): Promise<Job> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/jobs/${encodeURIComponent(jobId)}`);
  } catch {
    throw new ApiError('no connection', 0);
  }
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  return (await response.json()) as Job;
}

/** Whether uploads need an access code (the API's /health says so). */
export async function accessCodeRequired(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    const health = (await response.json()) as { access_code_required?: boolean };
    return health.access_code_required === true;
  } catch {
    return false; // can't tell; the upload itself will say if a code is needed
  }
}

const ACCESS_CODE_KEY = 'scan2sim.accessCode';

export function savedAccessCode(): string | null {
  try {
    return localStorage.getItem(ACCESS_CODE_KEY);
  } catch {
    return null;
  }
}

export function saveAccessCode(code: string | null): void {
  try {
    if (code) localStorage.setItem(ACCESS_CODE_KEY, code);
    else localStorage.removeItem(ACCESS_CODE_KEY);
  } catch {
    // Storage can be unavailable (private browsing); the code then lasts for this visit only.
  }
}

/**
 * Upload a video to start a job; resolves to the new job's id. Uses XMLHttpRequest because
 * fetch can't report upload progress, which matters for big phone videos.
 */
export function createJob(
  video: File,
  title: string,
  accessCode: string | null,
  onProgress: (fraction: number) => void,
): { done: Promise<string>; cancel: () => void } {
  const request = new XMLHttpRequest();
  const done = new Promise<string>((resolve, reject) => {
    request.open('POST', `${API_URL}/jobs`);
    if (accessCode) request.setRequestHeader('X-Access-Code', accessCode);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve((JSON.parse(request.responseText) as { job_id: string }).job_id);
        } catch {
          reject(new ApiError('unexpected answer from the server', request.status));
        }
      } else {
        reject(new ApiError(detailFromText(request.responseText) ?? `HTTP ${request.status}`, request.status));
      }
    };
    request.onerror = () => reject(new ApiError('no connection', 0));
    request.onabort = () => reject(new ApiError('cancelled', -1));
    const form = new FormData();
    form.append('video', video);
    form.append('title', title);
    request.send(form);
  });
  return { done, cancel: () => request.abort() };
}

export interface WhatIfAnswer {
  /** Proposed actions, unchecked: pass them through parseActions before using them. */
  actions: unknown;
  narration: string;
}

/** Ask Muse Spark how a plain-English scenario should play out in this scene. */
export async function askWhatIf(prompt: string, objects: unknown[], accessCode: string | null): Promise<WhatIfAnswer> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/what-if`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(accessCode ? { 'X-Access-Code': accessCode } : {}) },
      body: JSON.stringify({ prompt, objects }),
    });
  } catch {
    throw new ApiError('no connection', 0);
  }
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  const answer = (await response.json()) as { actions?: unknown; narration?: unknown };
  return { actions: answer.actions, narration: typeof answer.narration === 'string' ? answer.narration : '' };
}

async function errorDetail(response: Response): Promise<string> {
  return detailFromText(await response.text().catch(() => '')) ?? `HTTP ${response.status}`;
}

/** FastAPI errors look like {"detail": "..."}. */
function detailFromText(text: string): string | null {
  try {
    const detail = (JSON.parse(text) as { detail?: unknown }).detail;
    return typeof detail === 'string' ? detail : null;
  } catch {
    return null;
  }
}
