import ReactMarkdown, { type Components } from "react-markdown";
import { cn } from "@/lib/utils";

// No rehype-raw: raw HTML in catalog text is shown as text, never rendered.
const COMPONENTS: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-saffron underline underline-offset-2">
      {children}
    </a>
  ),
  code: ({ children }) => <code className="rounded bg-raised px-1 py-0.5 font-mono text-[13px]">{children}</code>,
  p: ({ children }) => <p className="leading-relaxed [&:not(:last-child)]:mb-2">{children}</p>,
  ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
  img: ({ alt }) => <span>{alt}</span>,
};

export function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("min-w-0 break-words", className)}>
      <ReactMarkdown components={COMPONENTS} skipHtml={false}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** Inline markdown for list items: no wrapping paragraph. */
export function InlineMarkdown({ text }: { text: string }) {
  return (
    <ReactMarkdown components={{ ...COMPONENTS, p: ({ children }) => <>{children}</> }}>{text}</ReactMarkdown>
  );
}
