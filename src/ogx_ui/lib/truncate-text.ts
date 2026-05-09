export function truncateText(
  text: string | null | undefined,
  maxLength: number = 50
): string {
  if (!text) return "N/A";
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
}
