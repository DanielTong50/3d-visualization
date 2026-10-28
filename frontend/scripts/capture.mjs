#!/usr/bin/env node
/**
 * Render the test data the vision models are evaluated on, straight from the plan library:
 *
 *   assemblies/<id>/manual/page-NN.png    every page of the generated manual, at print size
 *   assemblies/<id>/session/session.mp4   a clean session (no AR layer, no HUD), filmed handheld
 *   assemblies/<id>/session/truth.json    what is done at every instant of that video
 *
 * Usage: node scripts/capture.mjs [--manuals] [--sessions] [--only id,id] [--fps 15]
 * (no flag: both). Starts its own Vite server. Needs Playwright with Chrome or Chromium, and
 * ffmpeg for the sessions.
 */
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LIBRARY = join(ROOT, '..', 'assemblies');
const VIDEO = { width: 1280, height: 720 };

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const wantManuals = flag('manuals') || !flag('sessions');
const wantSessions = flag('sessions') || !flag('manuals');
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
  await page.close();

  // When each step is finished: the end of its verify phase.
  const steps = segments
    .filter((s) => s.phase === 'verify')
    .map((s) => ({ index: s.step, start: segments.find((x) => x.step === s.step).start, done_at: +(s.start + s.duration).toFixed(3) }));
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

const server = await createServer({ root: ROOT, logLevel: 'error', server: { port: 5299, strictPort: false } });
await server.listen();
const base = server.resolvedUrls.local[0];
const browser = await launch();
try {
  const library = JSON.parse(await readFile(join(LIBRARY, 'index.json'), 'utf8'));
  const only = option('only', '').split(',').filter(Boolean);
  for (const item of library.filter((p) => !only.length || only.includes(p.id))) {
    if (wantManuals) await manual(browser, base, item.id, 3 + item.steps);
    if (wantSessions) await session(browser, base, item.id);
  }
} finally {
  await browser.close();
  await server.close();
}
