"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Edit, Search, Trash2 } from "lucide-react";
import { Prompt, PromptFilters } from "./types";

interface PromptListProps {
  prompts: Prompt[];
  onEdit: (prompt: Prompt) => void;
  onDelete: (promptId: string) => void;
}

export function PromptList({ prompts, onEdit, onDelete }: PromptListProps) {
  const [filters, setFilters] = useState<PromptFilters>({});

  const filteredPrompts = prompts.filter(prompt => {
    if (
      filters.searchTerm &&
      !(
        prompt.prompt
          ?.toLowerCase()
          .includes(filters.searchTerm.toLowerCase()) ||
        prompt.prompt_id
          .toLowerCase()
          .includes(filters.searchTerm.toLowerCase())
      )
    ) {
      return false;
    }
    return true;
  });

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search prompts..."
            value={filters.searchTerm || ""}
            onChange={e =>
              setFilters(prev => ({ ...prev, searchTerm: e.target.value }))
            }
            className="pl-10"
          />
        </div>
      </div>

      {/* Prompts Table */}
      <div className="overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Content</TableHead>
              <TableHead>Variables</TableHead>
              <TableHead>Version</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredPrompts.map(prompt => (
              <TableRow key={prompt.prompt_id}>
                <TableCell className="max-w-48">
                  <Button
                    variant="link"
                    className="p-0 h-auto font-mono text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 max-w-full justify-start"
                    onClick={() => onEdit(prompt)}
                    title={prompt.prompt_id}
                  >
                    <div className="truncate">{prompt.prompt_id}</div>
                  </Button>
                </TableCell>
                <TableCell className="max-w-64">
                  <div
                    className="font-mono text-xs text-muted-foreground truncate"
                    title={prompt.prompt || "No content"}
                  >
                    {prompt.prompt || "No content"}
                  </div>
                </TableCell>
                <TableCell>
                  {prompt.variables.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {prompt.variables.map(variable => (
                        <Badge
                          key={variable}
                          variant="outline"
                          className="text-xs"
                        >
                          {variable}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground text-sm">None</span>
                  )}
                </TableCell>
                <TableCell className="text-sm">
                  {prompt.version}
                  {prompt.is_default && (
                    <Badge variant="secondary" className="text-xs ml-2">
                      Default
                    </Badge>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onEdit(prompt)}
                      className="h-8 w-8 p-0"
                    >
                      <Edit className="h-3 w-3" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        if (
                          confirm(
                            `Are you sure you want to delete this prompt? This action cannot be undone.`
                          )
                        ) {
                          onDelete(prompt.prompt_id);
                        }
                      }}
                      className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {filteredPrompts.length === 0 && (
        <div className="text-center py-12">
          <div className="text-muted-foreground">
            {prompts.length === 0
              ? "No prompts yet"
              : "No prompts match your filters"}
          </div>
        </div>
      )}
    </div>
  );
}
