import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { defineConfig, Plugin } from 'vite';

function apiPlugin(): Plugin {
  return {
    name: 'apple-support-api-plugin',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/api/')) {
          return next();
        }

        const parsedUrl = new URL(req.url, 'http://localhost:3000');
        const pathname = parsedUrl.pathname;

        // 1. GET /api/dataset
        if (pathname === '/api/dataset') {
          const filePath = path.resolve(__dirname, 'data', 'golden_eval_set.json');
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'application/json');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
          res.statusCode = 404;
          res.end(JSON.stringify({ error: 'golden_eval_set.json not found' }));
          return;
        }

        // 2. GET /api/benchmark
        if (pathname === '/api/benchmark') {
          const filePath = path.resolve(__dirname, 'data', 'evaluation_results.json');
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'application/json');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
          res.statusCode = 404;
          res.end(JSON.stringify({ error: 'evaluation_results.json not found' }));
          return;
        }

        // 3. GET /api/exemplars
        if (pathname === '/api/exemplars') {
          const filePath = path.resolve(__dirname, 'data', 'historical_exemplars.json');
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'application/json');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
          res.statusCode = 404;
          res.end(JSON.stringify({ error: 'historical_exemplars.json not found' }));
          return;
        }

        // 4. GET /api/report
        if (pathname === '/api/report') {
          const filePath = path.resolve(__dirname, 'REPORT.md');
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
          res.statusCode = 404;
          res.end('REPORT.md not found');
          return;
        }

        // 5. POST /api/query (Run live inference on tweet query)
        if (pathname === '/api/query' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => { body += chunk; });
          req.on('end', () => {
            try {
              const data = JSON.parse(body || '{}');
              const query = data.query || '';
              const useLlm = Boolean(data.llm);
              if (!query.trim()) {
                res.statusCode = 400;
                res.end(JSON.stringify({ error: 'Query text is required' }));
                return;
              }

              const args = ['scripts/run_eval.py', '--query', query, '--json'];
              if (useLlm) args.push('--llm');

              execFile('python3', args, { cwd: __dirname, timeout: 12000 }, (error, stdout, stderr) => {
                if (error) {
                  res.statusCode = 500;
                  res.end(JSON.stringify({ error: error.message, stderr }));
                  return;
                }
                res.setHeader('Content-Type', 'application/json');
                res.end(stdout.trim());
              });
            } catch (err: any) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: err.message }));
            }
          });
          return;
        }

        // 6. POST /api/run-eval (Rerun benchmark evaluation)
        if (pathname === '/api/run-eval' && req.method === 'POST') {
          execFile('python3', ['scripts/evaluator.py'], { cwd: __dirname, timeout: 30000 }, (error, stdout, stderr) => {
            if (error) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: error.message, stderr }));
              return;
            }
            const filePath = path.resolve(__dirname, 'data', 'evaluation_results.json');
            if (fs.existsSync(filePath)) {
              res.setHeader('Content-Type', 'application/json');
              fs.createReadStream(filePath).pipe(res);
              return;
            }
            res.end(JSON.stringify({ success: true, stdout }));
          });
          return;
        }

        next();
      });
    }
  };
}

// LINT.IfChange(aistudio_media_plugin)
function aistudioMediaPlugin(): Plugin {
  return {
    name: 'vite-plugin-aistudio-media',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/assets/aistudio/')) {
          const rawPath = req.url.split('?')[0].split('#')[0];
          try {
            const decodedPath = decodeURIComponent(rawPath);
            const relativePath = decodedPath.replace(/^\//, '');
            const aistudioDir = path.resolve(
              __dirname,
              'public',
              'assets',
              'aistudio',
            );
            const filePath = path.resolve(__dirname, 'public', relativePath);
            if (
              filePath.startsWith(aistudioDir + path.sep) &&
              fs.existsSync(filePath) &&
              fs.statSync(filePath).isFile()
            ) {
              const ext = path.extname(filePath).toLowerCase();
              const mimeMap: Record<string, string> = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.svg': 'image/svg+xml',
                '.bmp': 'image/bmp',
                '.ico': 'image/x-icon',
                '.mp4': 'video/mp4',
                '.webm': 'video/webm',
                '.ogv': 'video/ogg',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.ogg': 'audio/ogg',
                '.pdf': 'application/pdf',
              };
              res.setHeader(
                'Content-Type',
                mimeMap[ext] || 'application/octet-stream',
              );
              res.setHeader('Cache-Control', 'no-cache');
              fs.createReadStream(filePath).pipe(res);
              return;
            }
          } catch {
            // Fall through if URI decoding or file access fails
          }
        }
        next();
      });
    },
  };
}
// LINT.ThenChange(//depot/google3/java/com/google/alkali/boq/makersuite/applet_dev_service/templates/initializers/react_theme/vite.config.ts:aistudio_media_plugin)

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss(), aistudioMediaPlugin(), apiPlugin()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
