import React from "react";

export interface MessageBlockProps {
  label: string;
  labelDetail?: string;
  content: React.ReactNode;
  className?: string;
  contentClassName?: string;
}

export const MessageBlock: React.FC<MessageBlockProps> = ({
  label,
  labelDetail,
  content,
  className = "",
  contentClassName = "",
}) => {
  return (
    <div className={`mb-4 ${className}`}>
      <p className="py-1 font-semibold text-muted-foreground mb-1">
        {label}
        {labelDetail && (
          <span className="text-xs text-muted-foreground font-normal ml-1">
            {labelDetail}
          </span>
        )}
      </p>
      <div className={`py-1 whitespace-pre-wrap ${contentClassName}`}>
        {content}
      </div>
    </div>
  );
};

export interface ToolCallBlockProps {
  children: React.ReactNode;
  className?: string;
}

export const ToolCallBlock = ({ children, className }: ToolCallBlockProps) => {
  const baseClassName =
    "p-3 bg-slate-50 border border-slate-200 rounded-md text-sm";

  return (
    <div className={`${baseClassName} ${className || ""}`}>
      <pre className="whitespace-pre-wrap text-xs">{children}</pre>
    </div>
  );
};
