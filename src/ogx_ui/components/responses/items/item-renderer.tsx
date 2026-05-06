import {
  isMessageItem,
  isFunctionCallItem,
  isWebSearchCallItem,
  AnyResponseItem,
} from "../utils/item-types";
import { MessageItemComponent } from "./message-item";
import { FunctionCallItemComponent } from "./function-call-item";
import { WebSearchItemComponent } from "./web-search-item";
import { GenericItemComponent } from "./generic-item";

interface ItemRendererProps {
  item: AnyResponseItem;
  index: number;
  keyPrefix: string;
  defaultRole?: string;
}

export function ItemRenderer({
  item,
  index,
  keyPrefix,
  defaultRole = "unknown",
}: ItemRendererProps) {
  if (isMessageItem(item)) {
    return (
      <MessageItemComponent
        item={item}
        index={index}
        keyPrefix={keyPrefix}
        defaultRole={defaultRole}
      />
    );
  }

  if (isFunctionCallItem(item)) {
    return (
      <FunctionCallItemComponent
        item={item}
        index={index}
        keyPrefix={keyPrefix}
      />
    );
  }

  if (isWebSearchCallItem(item)) {
    return (
      <WebSearchItemComponent item={item} index={index} keyPrefix={keyPrefix} />
    );
  }

  // Fallback to generic item for unknown types
  return (
    <GenericItemComponent
      item={item as Record<string, unknown>}
      index={index}
      keyPrefix={keyPrefix}
    />
  );
}
