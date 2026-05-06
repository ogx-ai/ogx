import { MessageBlock } from "@/components/chat-playground/message-components";
import { MessageItem } from "../utils/item-types";

interface MessageItemProps {
  item: MessageItem;
  index: number;
  keyPrefix: string;
  defaultRole?: string;
}

export function MessageItemComponent({
  item,
  index,
  keyPrefix,
  defaultRole = "unknown",
}: MessageItemProps) {
  let content = "";

  if (typeof item.content === "string") {
    content = item.content;
  } else if (Array.isArray(item.content)) {
    content = item.content
      .map(c => {
        return c.type === "input_text" || c.type === "output_text"
          ? c.text
          : JSON.stringify(c);
      })
      .join(" ");
  }

  const role = item.role || defaultRole;
  const label = role.charAt(0).toUpperCase() + role.slice(1);

  return (
    <MessageBlock
      key={`${keyPrefix}-${index}`}
      label={label}
      content={content}
    />
  );
}
