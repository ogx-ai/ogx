#!/usr/bin/env node
// Copyright (c) The OGX Contributors.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/**
 * Test runner that finds and executes TypeScript tests based on suite/setup mapping.
 * Called by integration-tests.sh via npm test.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const suite = process.env.OGX_TEST_SUITE;
const setup = process.env.OGX_TEST_SETUP || '';

if (!suite) {
  console.error('Error: OGX_TEST_SUITE environment variable is required');
  process.exit(1);
}

// Read suites.json to find matching test files
const suitesPath = path.join(__dirname, 'suites.json');
if (!fs.existsSync(suitesPath)) {
  console.log(`No TypeScript tests configured (${suitesPath} not found)`);
  process.exit(0);
}

const suites = JSON.parse(fs.readFileSync(suitesPath, 'utf-8'));

// Find matching entry
let testFiles = [];
for (const entry of suites) {
  if (entry.suite !== suite) {
    continue;
  }
  const entrySetup = entry.setup || '';
  if (entrySetup && entrySetup !== setup) {
    continue;
  }
  testFiles = entry.files || [];
  break;
}

if (testFiles.length === 0) {
  console.log(`No TypeScript integration tests mapped for suite ${suite} (setup ${setup})`);
  process.exit(0);
}

console.log(`Running TypeScript tests for suite ${suite} (setup ${setup}): ${testFiles.join(', ')}`);

// Run Jest with the mapped test files
try {
  execSync(`npx jest --config jest.integration.config.js ${testFiles.join(' ')}`, {
    stdio: 'inherit',
    cwd: __dirname,
  });
} catch (error) {
  process.exit(error.status || 1);
}
