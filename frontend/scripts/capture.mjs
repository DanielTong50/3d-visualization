#!/usr/bin/env node
/**
 * Render the test data the vision models are evaluated on, straight from the plan library:
 *
 *   assemblies/<id>/manual/page-NN.png    every page of the generated manual, at print size
 *   assemblies/<id>/session/session.mp4   a clean session (no AR layer, no HUD), filmed handheld
 *   assemblies/<id>/session/truth.json    what is done at every instant of that video
 *   assemblies/<id>/session/ids.bin.gz    exact part label maps (5 fps, 320x180) for scoring part tracking
 *   public/posters/copilot*.jpg           stills of the copilot session for the home page
 *
 * Usage: node scripts/capture.mjs [--manuals] [--sessions] [--posters] [--only id,id] [--fps 15]
 * (no flag: manuals and sessions). Starts its own Vite server. Needs Playwright with Chrome or
 * Chromium, and ffmpeg for the sessions.
 */
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LIBRARY = join(ROOT, '..', 'assemblies');
const VIDEO = { width: 1280, height: 720 };
const IDS_FPS = 5;
const IDS_W = 320;
const IDS_H = 180;

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const chosen = ['manuals', 'sessions', 'posters'].filter(flag);
const wantManuals = flag('manuals') || !chosen.length;
const wantSessions = flag('sessions') || !chosen.length;
const wantPosters = flag('posters');
const fps = Number(option('fps', '15'));

async function launch() {
  try {
    return await chromium.launch({ channel: 'chrome', args: ['--use-angle=metal', '--ignore-gpu-blocklist'] });
  } catch {
    return chromium.launch();
  }
}

async function manual(browser, base, id, pages) {
  const out = join(LIBRARY, id, 'manual');
  await mkdir(out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1240, height: 1754 } });
  for (let n = 1; n <= pages; n++) {
    await page.goto(`${base}?view=render&furniture=${id}&page=${n}`);
    await page.waitForSelector('html[data-ready="true"] .mp');
    await page.waitForTimeout(150);
    await page.locator('.mp').screenshot({ path: join(out, `page-${String(n).padStart(2, '0')}.png`) });
  }
  await page.close();
  console.log(`${id}: ${pages} manual pages`);
}

function encoder(path) {
  const ffmpeg = spawn(
    'ffmpeg',
    [
      '-y',
      '-loglevel',
      'error',
      '-f',
      'image2pipe',
      '-framerate',
      String(fps),
      '-c:v',
      'mjpeg',
      '-i',
      '-',
      '-c:v',
      'libx264',
      '-pix_fmt',
      'yuv420p',
      '-crf',
      '27',
      '-preset',
      'slow',
      '-movflags',
      '+faststart',
      path,
    ],
    { stdio: ['pipe', 'inherit', 'inherit'] },
  );
  const done = new Promise((resolve, reject) =>
    ffmpeg.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`)))),
  );
  return { ffmpeg, done };
}

async function session(browser, base, id) {
  const out = join(LIBRARY, id, 'session');
  await mkdir(out, { recursive: true });
  const page = await browser.newPage({ viewport: VIDEO });
  await page.goto(`${base}?view=render&furniture=${id}`);
  await page.waitForSelector('html[data-ready="true"]');
  const { duration, segments } = await page.evaluate(() => ({ duration: window.__darius.duration, segments: window.__darius.segments }));
  const frames = Math.ceil(duration * fps);
  const { ffmpeg, done } = encoder(join(out, 'session.mp4'));
  const states = [];
  for (let i = 0; i < frames; i++) {
    const t = i / fps;
    states.push(await page.evaluate((t) => window.__darius.draw(t), t));
    const jpeg = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!ffmpeg.stdin.write(jpeg)) await new Promise((resolve) => ffmpeg.stdin.once('drain', resolve));
  }
  ffmpeg.stdin.end();
  await done;

  // When each step is finished: the end of its verify phase.
  const steps = segments
    .filter((s) => s.phase === 'verify')
    .map((s) => ({ index: s.step, start: segments.find((x) => x.step === s.step).start, done_at: +(s.start + s.duration).toFixed(3) }));
  // Ground-truth part masks for scoring part tracking: label maps at IDS_FPS, IDS_W x IDS_H.
  const labels = [];
  let parts = [];
  for (let i = 0; i < Math.floor(duration * IDS_FPS); i++) {
    const frame = await page.evaluate(({ t, w, h }) => window.__darius.ids(t, w, h), { t: i / IDS_FPS, w: IDS_W, h: IDS_H });
    labels.push(Buffer.from(frame.labels, 'base64'));
    parts = frame.parts;
  }
  await writeFile(join(out, 'ids.bin.gz'), gzipSync(Buffer.concat(labels), { level: 9 }));
  await writeFile(
    join(out, 'ids.json'),
    JSON.stringify({
      fps: IDS_FPS,
      width: IDS_W,
      height: IDS_H,
      frames: labels.length,
      parts,
      encoding: 'uint8 label = part index + 1, gzip',
    }) + '\n',
  );
  await page.close();

  const truth = {
    furniture: id,
    fps,
    duration: +duration.toFixed(3),
    frames,
    size: [VIDEO.width, VIDEO.height],
    steps,
    segments: segments.map((s) => ({ ...s, start: +s.start.toFixed(3), duration: +s.duration.toFixed(3) })),
    completed: states.map((s) => s.completed),
  };
  await writeFile(join(out, 'truth.json'), JSON.stringify(truth) + '\n');
  console.log(`${id}: ${frames} frames, ${duration.toFixed(1)} s`);
}

/** The copilot mid-step (ghosts, arcs, tracked parts, the HUD), paused at time t. */
async function poster(browser, base, id, t, name) {
  const out = join(ROOT, 'public', 'posters');
  await mkdir(out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.goto(`${base}?view=ar&furniture=${id}`);
  await page.waitForSelector('.scrub input');
  await page.waitForTimeout(1200);
  await page.click('button[aria-label="Pause"]');
  await page.evaluate((t) => {
    const input = document.querySelector('.scrub input');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, String(t));
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, t);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(out, `${name}.jpg`), type: 'jpeg', quality: 84 });
  await page.close();
  console.log(`poster ${name}: ${id} at ${t} s`);
}

const server = await createServer({ root: ROOT, logLevel: 'error', server: { port: 5299, strictPort: false } });
await server.listen();
const base = server.resolvedUrls.local[0];
const browser = await launch();
try {
  const library = JSON.parse(await readFile(join(LIBRARY, 'index.json'), 'utf8'));
  const only = option('only', '').split(',').filter(Boolean);
  if (wantPosters) {
    await poster(browser, base, 'halden-bookcase', 5.1, 'copilot'); // parts flying to their ghosts
    await poster(browser, base, 'halden-bookcase', 9.2, 'copilot-fasten'); // fittings going in, with the close-up
    await poster(browser, base, 'halden-bookcase', 47.6, 'copilot-turn'); // standing the bookcase up
  }
  for (const item of library.filter((p) => !only.length || only.includes(p.id))) {
    if (wantManuals) await manual(browser, base, item.id, 3 + item.steps);
    if (wantSessions) await session(browser, base, item.id);
  }
} finally {
  await browser.close();
  await server.close();
}
