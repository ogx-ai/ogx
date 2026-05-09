#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const port = process.env.OGX_UI_PORT || 8322;
const uiDir = path.resolve(__dirname, '..');
const serverPath = path.join(uiDir, '.next', 'standalone', 'ui', 'src', 'ogx_ui', 'server.js');
const serverDir = path.dirname(serverPath);

console.log(`Starting OGX UI on http://localhost:${port}`);

const child = spawn(process.execPath, [serverPath], {
  cwd: serverDir,
  stdio: 'inherit',
  env: {
    ...process.env,
    PORT: port,
  },
});

process.on('SIGINT', () => {
  child.kill('SIGINT');
  process.exit(0);
});

process.on('SIGTERM', () => {
  child.kill('SIGTERM');
  process.exit(0);
});

child.on('exit', (code) => {
  process.exit(code);
});
