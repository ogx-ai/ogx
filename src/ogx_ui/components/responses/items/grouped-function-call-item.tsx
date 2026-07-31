import {
  MessageBlock,
  ToolCallBlock,
} from "@/components/chat-playground/message-components";
import { FunctionCallItem, FunctionCallOutputItem } from "../utils/item-types";

interface GroupedFunctionCallItemProps {
  functionCall: FunctionCallItem;
  output: FunctionCallOutputItem;
  index: number;
  keyPrefix: string;
}

export function GroupedFunctionCallItemComponent({
  functionCall,
  output,
  index,
  keyPrefix,
}: GroupedFunctionCallItemProps) {
  const name = functionCall.name || "unknown";
  const args = functionCall.arguments || "{}";

  // Extract the output content from function_call_output
  let outputContent = "";
  if (output.output) {
    outputContent =
      typeof output.output === "string"
        ? output.output
        : JSON.stringify(output.output);
  } else {
    outputContent = JSON.stringify(output, null, 2);
  }

  const functionCallContent = (
    <div>
      <div className="mb-2">
        <span className="text-sm text-gray-600">Arguments</span>
        <ToolCallBlock>{`${name}(${args})`}</ToolCallBlock>
      </div>
      <div>
        <span className="text-sm text-gray-600">Output</span>
        <ToolCallBlock>{outputContent}</ToolCallBlock>
      </div>
    </div>
  );

  return (
    <MessageBlock
      key={`${keyPrefix}-${index}`}
      label="Function Call"
      content={functionCallContent}
    />
  );
}
