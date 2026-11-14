import { useEffect } from "react";

/**
 * Document-level ArrowLeft / ArrowRight paging for image list views, shared by
 * the history sample gallery and the live-monitor sample preview. Disabled
 * while a modal owns navigation (the image viewer dialog has its own arrow-key
 * handling) and inside editable targets so paging never fights typing.
 */
export function useArrowKeyPaging({ enabled = true, canPrev, canNext, onPrev, onNext }: {
  enabled?: boolean;
  canPrev: boolean;
  canNext: boolean;
  onPrev: () => void;
  onNext: () => void;
}) {
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest("input, textarea, select"))) return;
      if (event.key === "ArrowLeft" && canPrev) {
        event.preventDefault();
        onPrev();
      } else if (event.key === "ArrowRight" && canNext) {
        event.preventDefault();
        onNext();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [canNext, canPrev, enabled, onPrev, onNext]);
}
