import {
  MessageBlock,
  ToolCallBlock,
} from "@/components/chat-playground/message-components";
import { BaseItem } from "../utils/item-types";

interface GenericItemProps {
  item: BaseItem;
  index: number;
  keyPrefix: string;
}

export function GenericItemComponent({
  item,
  index,
  keyPrefix,
}: GenericItemProps) {
  // Handle other types like function calls, tool outputs, etc.
  const itemData = item as Record<string, unknown>;

  const content = itemData.content
    ? typeof itemData.content === "string"
      ? itemData.content
      : JSON.stringify(itemData.content, null, 2)
    : JSON.stringify(itemData, null, 2);

  const label = keyPrefix === "input" ? "Input" : "Output";

  return (
    <MessageBlock
      key={`${keyPrefix}-${index}`}
      label={label}
      labelDetail={`(${itemData.type})`}
      content={<ToolCallBlock>{content}</ToolCallBlock>}
    />
  );
}
