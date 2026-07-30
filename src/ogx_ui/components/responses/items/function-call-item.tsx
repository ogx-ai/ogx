import {
  MessageBlock,
  ToolCallBlock,
} from "@/components/chat-playground/message-components";
import { FunctionCallItem } from "../utils/item-types";

interface FunctionCallItemProps {
  item: FunctionCallItem;
  index: number;
  keyPrefix: string;
}

export function FunctionCallItemComponent({
  item,
  index,
  keyPrefix,
}: FunctionCallItemProps) {
  const name = item.name || "unknown";
  const args = item.arguments || "{}";
  const formattedFunctionCall = `${name}(${args})`;

  return (
    <MessageBlock
      key={`${keyPrefix}-${index}`}
      label="Function Call"
      content={<ToolCallBlock>{formattedFunctionCall}</ToolCallBlock>}
    />
  );
}
