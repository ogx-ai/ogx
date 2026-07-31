"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { X, Plus, Save, Trash2 } from "lucide-react";
import { Prompt, PromptFormData } from "./types";

interface PromptEditorProps {
  prompt?: Prompt;
  onSave: (prompt: PromptFormData) => void;
  onCancel: () => void;
  onDelete?: (promptId: string) => void;
  error?: string | null;
}

export function PromptEditor({
  prompt,
  onSave,
  onCancel,
  onDelete,
  error,
}: PromptEditorProps) {
  const [formData, setFormData] = useState<PromptFormData>({
    prompt: "",
    variables: [],
  });

  const [newVariable, setNewVariable] = useState("");
  const [variableValues, setVariableValues] = useState<Record<string, string>>(
    {}
  );

  useEffect(() => {
    if (prompt) {
      setFormData({
        prompt: prompt.prompt || "",
        variables: prompt.variables || [],
      });
    }
  }, [prompt]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.prompt.trim()) {
      return;
    }
    onSave(formData);
  };

  const addVariable = () => {
    if (
      newVariable.trim() &&
      !formData.variables.includes(newVariable.trim())
    ) {
      setFormData(prev => ({
        ...prev,
        variables: [...prev.variables, newVariable.trim()],
      }));
      setNewVariable("");
    }
  };

  const removeVariable = (variableToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      variables: prev.variables.filter(
        variable => variable !== variableToRemove
      ),
    }));
  };

  const renderPreview = () => {
    const text = formData.prompt;
    if (!text) return text;

    // Split text by variable patterns and process each part
    const parts = text.split(/(\{\{\s*\w+\s*\}\})/g);

    return parts.map((part, index) => {
      const variableMatch = part.match(/\{\{\s*(\w+)\s*\}\}/);
      if (variableMatch) {
        const variableName = variableMatch[1];
        const isDefined = formData.variables.includes(variableName);
        const value = variableValues[variableName];

        if (!isDefined) {
          // Variable not in variables list - likely a typo/bug (RED)
          return (
            <span
              key={index}
              className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 px-1 rounded font-medium"
            >
              {part}
            </span>
          );
        } else if (value && value.trim()) {
          // Variable defined and has value - show the value (GREEN)
          return (
            <span
              key={index}
              className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 px-1 rounded font-medium"
            >
              {value}
            </span>
          );
        } else {
          // Variable defined but empty (YELLOW)
          return (
            <span
              key={index}
              className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 px-1 rounded font-medium"
            >
              {part}
            </span>
          );
        }
      }
      return part;
    });
  };

  const updateVariableValue = (variable: string, value: string) => {
    setVariableValues(prev => ({
      ...prev,
      [variable]: value,
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{error}</p>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form Section */}
        <div className="space-y-4">
          <div>
            <Label htmlFor="prompt">Prompt Content *</Label>
            <Textarea
              id="prompt"
              value={formData.prompt}
              onChange={e =>
                setFormData(prev => ({ ...prev, prompt: e.target.value }))
              }
              placeholder="Enter your prompt content here. Use {{variable_name}} for dynamic variables."
              className="min-h-32 font-mono mt-2"
              required
            />
            <p className="text-xs text-muted-foreground mt-2">
              Use double curly braces around variable names, e.g.,{" "}
              {`{{user_name}}`} or {`{{topic}}`}
            </p>
          </div>

          <div className="space-y-3">
            <Label className="text-sm font-medium">Variables</Label>

            <div className="flex gap-2 mt-2">
              <Input
                value={newVariable}
                onChange={e => setNewVariable(e.target.value)}
                placeholder="Add variable name (e.g. user_name, topic)"
                onKeyPress={e =>
                  e.key === "Enter" && (e.preventDefault(), addVariable())
                }
                className="flex-1"
              />
              <Button
                type="button"
                onClick={addVariable}
                size="sm"
                disabled={
                  !newVariable.trim() ||
                  formData.variables.includes(newVariable.trim())
                }
              >
                <Plus className="h-4 w-4" />
                Add
              </Button>
            </div>

            {formData.variables.length > 0 && (
              <div className="border rounded-lg p-3 bg-muted/20">
                <div className="flex flex-wrap gap-2">
                  {formData.variables.map(variable => (
                    <Badge
                      key={variable}
                      variant="secondary"
                      className="text-sm px-2 py-1"
                    >
                      {variable}
                      <button
                        type="button"
                        onClick={() => removeVariable(variable)}
                        className="ml-2 hover:text-destructive transition-colors"
                        title={`Remove ${variable} variable`}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <p className="text-xs text-muted-foreground">
              Variables that can be used in the prompt template. Each variable
              should match a {`{{variable}}`} placeholder in the content above.
            </p>
          </div>
        </div>

        {/* Preview Section */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Preview</CardTitle>
              <CardDescription>
                Live preview of compiled prompt and variable substitution.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {formData.prompt ? (
                <>
                  {/* Variable Values */}
                  {formData.variables.length > 0 && (
                    <div className="space-y-3">
                      <Label className="text-sm font-medium">
                        Variable Values
                      </Label>
                      <div className="space-y-2">
                        {formData.variables.map(variable => (
                          <div
                            key={variable}
                            className="grid grid-cols-2 gap-3 items-center"
                          >
                            <div className="text-sm font-mono text-muted-foreground">
                              {variable}
                            </div>
                            <Input
                              id={`var-${variable}`}
                              value={variableValues[variable] || ""}
                              onChange={e =>
                                updateVariableValue(variable, e.target.value)
                              }
                              placeholder={`Enter value for ${variable}`}
                              className="text-sm"
                            />
                          </div>
                        ))}
                      </div>
                      <Separator />
                    </div>
                  )}

                  {/* Live Preview */}
                  <div>
                    <Label className="text-sm font-medium mb-2 block">
                      Compiled Prompt
                    </Label>
                    <div className="bg-muted/50 p-4 rounded-lg border">
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">
                        {renderPreview()}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-4 mt-2 text-xs">
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-green-500 dark:bg-green-400 border rounded"></div>
                        <span className="text-muted-foreground">Used</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-yellow-500 dark:bg-yellow-400 border rounded"></div>
                        <span className="text-muted-foreground">Unused</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-red-500 dark:bg-red-400 border rounded"></div>
                        <span className="text-muted-foreground">Undefined</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8">
                  <div className="text-muted-foreground text-sm">
                    Enter content to preview the compiled prompt
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Use {`{{variable_name}}`} to add dynamic variables
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Separator />

      <div className="flex justify-between">
        <div>
          {prompt && onDelete && (
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                if (
                  confirm(
                    `Are you sure you want to delete this prompt? This action cannot be undone.`
                  )
                ) {
                  onDelete(prompt.prompt_id);
                }
              }}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete Prompt
            </Button>
          )}
        </div>
        <div className="flex gap-2">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">
            <Save className="h-4 w-4 mr-2" />
            {prompt ? "Update" : "Create"} Prompt
          </Button>
        </div>
      </div>
    </form>
  );
}
