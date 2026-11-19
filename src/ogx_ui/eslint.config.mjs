import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";
import prettierRecommended from "eslint-plugin-prettier/recommended";

export default [
  {
    // Build helpers and the linter config itself were never covered by
    // `next lint`, so they stay out of scope.
    ignores: ["bin/**", "scripts/**", "eslint.config.mjs"],
  },
  ...nextCoreWebVitals,
  ...nextTypescript,
  prettierRecommended,
  {
    // New strict rules introduced by eslint-config-next 16 (react-hooks v6,
    // typescript-eslint 8). Keep them as warnings until the existing code
    // is cleaned up, then promote them back to errors.
    rules: {
      "react-hooks/exhaustive-deps": "warn",
      "react-hooks/immutability": "warn",
      "react-hooks/preserve-manual-memoization": "warn",
      "react-hooks/purity": "warn",
      "react-hooks/refs": "warn",
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/static-components": "warn",
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": "warn",
    },
  },
];
