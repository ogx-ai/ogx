/**
 * Next.js Instrumentation
 * This file is used for initializing monitoring, tracing, or other observability tools.
 * It runs once when the server starts, before any application code.
 *
 * Learn more: https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */

export async function register() {
  await import("./lib/config-validator");
}
