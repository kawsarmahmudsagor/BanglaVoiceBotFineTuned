"use client";

import { Dialog } from "@base-ui/react/dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";

/**
 * A right-side sheet. Its content stays mounted while closed, so work inside it
 * (for example a running upload) keeps going and shows its progress on reopen.
 */
export function Drawer({
  open,
  onOpenChange,
  title,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={(o) => onOpenChange(o)}>
      <Dialog.Portal keepMounted>
        <Dialog.Backdrop className="fixed inset-0 z-40 bg-ink/70 transition-opacity duration-200 data-[ending-style]:opacity-0 data-[starting-style]:opacity-0 data-[closed]:hidden" />
        <Dialog.Popup className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[560px] flex-col border-l border-border bg-card transition-transform duration-200 data-[ending-style]:translate-x-full data-[starting-style]:translate-x-full data-[closed]:hidden">
          <div className="flex items-center justify-between gap-2 px-5 pt-4">
            <Dialog.Title className="text-xl font-semibold">{title}</Dialog.Title>
            <Dialog.Close
              aria-label="Close panel"
              className="inline-flex size-9 items-center justify-center rounded-md text-muted-foreground hover:bg-raised hover:text-foreground"
            >
              <X className="size-4" aria-hidden />
            </Dialog.Close>
          </div>
          <div className="mx-5 mt-2 h-0.5 bg-rule" aria-hidden />
          <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
