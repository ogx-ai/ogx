import { test, expect } from "@playwright/test";

test.describe("LogsTable Scroll and Progressive Loading", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the chat completions page
    await page.goto("/logs/chat-completions");

    // Wait for initial data to load
    await page.waitForSelector("table tbody tr", { timeout: 10000 });
  });

  test("should progressively load more data to fill tall viewports", async ({
    page,
  }) => {
    // Set a tall viewport (1400px height)
    await page.setViewportSize({ width: 1200, height: 1400 });

    // Wait for the table to be visible
    await page.waitForSelector("table");

    // Wait a bit for progressive loading to potentially trigger
    await page.waitForTimeout(3000);

    // Count the number of rows loaded
    const rowCount = await page.locator("table tbody tr").count();

    // With a 1400px viewport, we should have more than the default 20 rows
    // Assuming each row is ~50px, we should fit at least 25-30 rows
    expect(rowCount).toBeGreaterThan(20);
  });

  test("should trigger infinite scroll when scrolling near bottom", async ({
    page,
  }) => {
    // Set a medium viewport
    await page.setViewportSize({ width: 1200, height: 800 });

    // Wait for initial load
    await page.waitForSelector("table tbody tr");

    // Get initial row count
    const initialRowCount = await page.locator("table tbody tr").count();

    // Find the scrollable container
    const scrollContainer = page.locator("div.overflow-auto").first();

    // Scroll to near the bottom
    await scrollContainer.evaluate(element => {
      element.scrollTop = element.scrollHeight - element.clientHeight - 100;
    });

    // Wait for loading indicator or new data
    await page.waitForTimeout(2000);

    // Check if more rows were loaded
    const newRowCount = await page.locator("table tbody tr").count();

    // We should have more rows after scrolling
    expect(newRowCount).toBeGreaterThan(initialRowCount);
  });
});
