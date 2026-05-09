/**
 * Formats a tool_call object into a string representation.
 * Example: "functionName(argumentsString)"
 * @param toolCall The tool_call object, expected to have a `function` property
 *                 with `name` and `arguments`.
 * @returns A formatted string or an empty string if data is malformed.
 */
export function formatToolCallToString(toolCall: {
  function?: { name?: string; arguments?: unknown };
}): string {
  if (
    !toolCall ||
    !toolCall.function ||
    typeof toolCall.function.name !== "string" ||
    toolCall.function.arguments === undefined
  ) {
    return "";
  }

  const name = toolCall.function.name;
  const args = toolCall.function.arguments;
  let argsString: string;

  if (typeof args === "string") {
    argsString = args;
  } else {
    try {
      argsString = JSON.stringify(args);
    } catch {
      return "";
    }
  }

  return `${name}(${argsString})`;
}
