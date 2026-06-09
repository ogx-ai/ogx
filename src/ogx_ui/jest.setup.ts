// Import ogx-client shims for Node environment
import "ogx-client/shims/node";

// Add any other global test setup here
import "@testing-library/jest-dom";

// Mock ResizeObserver globally
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock IntersectionObserver globally
global.IntersectionObserver = class IntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {}
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
} as any;
