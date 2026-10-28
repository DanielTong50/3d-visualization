import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const ASSEMBLIES_DIR = fileURLToPath(new URL('../assemblies', import.meta.url));

const CONTENT_TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.mp4': 'video/mp4',
};

/**
 * Assembly plans (and their rendered manuals and videos) live in the repo's /assemblies folder,
 * written by the engine (`darius export`). Serve them at /assemblies/... in development and copy
 * them into the build.
 */
function assembliesFolder(): Plugin {
  return {
    name: 'darius-assemblies-folder',
    configureServer(server) {
      server.middlewares.use('/assemblies', (req, res, next) => {
        let file: string;
        try {
          file = path.join(ASSEMBLIES_DIR, decodeURIComponent((req.url ?? '/').split('?')[0]));
        } catch {
          next();
          return;
        }
        if (!file.startsWith(ASSEMBLIES_DIR + path.sep) || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
          next();
          return;
        }
        res.setHeader('Content-Type', CONTENT_TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
        fs.createReadStream(file).pipe(res);
      });
    },
    writeBundle(options) {
      if (options.dir) fs.cpSync(ASSEMBLIES_DIR, path.join(options.dir, 'assemblies'), { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [react(), assembliesFolder()],
});
