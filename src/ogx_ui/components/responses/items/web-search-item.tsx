import {
  MessageBlock,
  ToolCallBlock,
} from "@/components/chat-playground/message-components";
import { WebSearchCallItem } from "../utils/item-types";

interface WebSearchItemProps {
  item: WebSearchCallItem;
  index: number;
  keyPrefix: string;
}

export function WebSearchItemComponent({
  item,
  index,
  keyPrefix,
}: WebSearchItemProps) {
  const formattedWebSearch = `web_search_call(status: ${item.status})`;

  return (
    <MessageBlock
      key={`${keyPrefix}-${index}`}
      label="Function Call"
      labelDetail="(Web Search)"
      content={<ToolCallBlock>{formattedWebSearch}</ToolCallBlock>}
    />
  );
}
