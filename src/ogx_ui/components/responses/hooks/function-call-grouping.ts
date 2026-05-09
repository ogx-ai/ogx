import { useMemo } from "react";
import {
  isFunctionCallOutputItem,
  AnyResponseItem,
  FunctionCallOutputItem,
} from "../utils/item-types";

export interface GroupedItem {
  item: AnyResponseItem;
  index: number;
  outputItem?: AnyResponseItem;
  outputIndex?: number;
}

/**
 * Hook to group function calls with their corresponding outputs
 * @param items Array of items to group
 * @returns Array of grouped items with their outputs
 */
export function useFunctionCallGrouping(
  items: AnyResponseItem[]
): GroupedItem[] {
  return useMemo(() => {
    const groupedItems: GroupedItem[] = [];
    const processedIndices = new Set<number>();

    // Build a map of call_id to indices for function_call_output items
    const callIdToIndices = new Map<string, number[]>();

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (isFunctionCallOutputItem(item)) {
        if (!callIdToIndices.has(item.call_id)) {
          callIdToIndices.set(item.call_id, []);
        }
        callIdToIndices.get(item.call_id)!.push(i);
      }
    }

    // Process items and group function calls with their outputs
    for (let i = 0; i < items.length; i++) {
      if (processedIndices.has(i)) {
        continue;
      }

      const currentItem = items[i];

      if (
        currentItem.type === "function_call" &&
        "name" in currentItem &&
        "call_id" in currentItem
      ) {
        const functionCallId = currentItem.call_id as string;
        let outputIndex = -1;
        let outputItem: FunctionCallOutputItem | null = null;

        const relatedIndices = callIdToIndices.get(functionCallId) || [];
        for (const idx of relatedIndices) {
          const potentialOutput = items[idx];
          outputIndex = idx;
          outputItem = potentialOutput as FunctionCallOutputItem;
          break;
        }

        if (outputItem && outputIndex !== -1) {
          // Group function call with its function_call_output
          groupedItems.push({
            item: currentItem,
            index: i,
            outputItem,
            outputIndex,
          });

          // Mark both items as processed
          processedIndices.add(i);
          processedIndices.add(outputIndex);

          // Matching function call and output found, skip to next item
          continue;
        }
      }
      // render normally
      groupedItems.push({
        item: currentItem,
        index: i,
      });
      processedIndices.add(i);
    }

    return groupedItems;
  }, [items]);
}
