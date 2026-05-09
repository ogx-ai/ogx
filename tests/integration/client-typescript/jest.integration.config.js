// Copyright (c) The OGX Contributors.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: {
          module: 'ES2022',
          moduleResolution: 'bundler',
        },
      },
    ],
  },
  testMatch: ['<rootDir>/__tests__/**/*.test.ts'],
  setupFilesAfterEnv: ['<rootDir>/setup.ts'],
  testTimeout: 60000, // 60 seconds (integration tests can be slow)
  watchman: false, // Disable watchman to avoid permission issues
};
