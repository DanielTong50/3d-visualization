import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, searchForWorkspaceRoot, type Plugin } from 'vite';

const SCENES_DIR = fileURLToPath(new URL('../scenes', import.meta.url));
// The materials table is shared with the pipeline; the dev server may read only that folder of it.
const SHARED_DATA_DIR = fileURLToPath(new URL('../pipeline/scan2sim/data', import.meta.url));

const CONTENT_TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.obj': 'text/plain',
  '.ply': 'application/octet-stream',
  '.splat': 'application/octet-stream',
  '.glb': 'model/gltf-binary',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
};

/**
 * Scene packages live in the repo's /scenes folder, shared with the pipeline.
 * Serve them at /scenes/... during development and copy them into the build.
 */
function scenesFolder(): Plugin {
  return {
    name: 'scan2sim-scenes-folder',
    configureServer(server) {
      server.middlewares.use('/scenes', (req, res, next) => {
        let file: string;
        try {
          file = path.join(SCENES_DIR, decodeURIComponent((req.url ?? '/').split('?')[0]));
        } catch {
          next();
          return;
        }
        if (!file.startsWith(SCENES_DIR + path.sep) || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
          next();
          return;
        }
        res.setHeader('Content-Type', CONTENT_TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
        fs.createReadStream(file).pipe(res);
      });
    },
    writeBundle(options) {
      if (options.dir) fs.cpSync(SCENES_DIR, path.join(options.dir, 'scenes'), { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [react(), scenesFolder()],
  server: { fs: { allow: [searchForWorkspaceRoot(process.cwd()), SHARED_DATA_DIR] } },
});
