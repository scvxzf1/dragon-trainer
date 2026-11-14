import { cleanup, fireEvent, renderHook } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { useArrowKeyPaging } from "./useArrowKeyPaging";

afterEach(() => {
  cleanup();
  document.querySelectorAll("input[data-test-target]").forEach((node) => node.remove());
});

it("pages on arrow keys, stops at both ends, and ignores typing targets", async () => {
  const user = userEvent.setup();
  const onPrev = vi.fn();
  const onNext = vi.fn();
  const { rerender } = renderHook((props: Parameters<typeof useArrowKeyPaging>[0]) => useArrowKeyPaging(props), {
    initialProps: { enabled: true, canPrev: false, canNext: true, onPrev, onNext },
  });

  await user.keyboard("{ArrowLeft}");
  expect(onPrev).not.toHaveBeenCalled();
  await user.keyboard("{ArrowRight}");
  expect(onNext).toHaveBeenCalledTimes(1);

  rerender({ enabled: true, canPrev: true, canNext: false, onPrev, onNext });
  await user.keyboard("{ArrowRight}");
  expect(onNext).toHaveBeenCalledTimes(1);
  await user.keyboard("{ArrowLeft}");
  expect(onPrev).toHaveBeenCalledTimes(1);

  rerender({ enabled: true, canPrev: true, canNext: true, onPrev, onNext });
  const input = document.createElement("input");
  input.setAttribute("data-test-target", "");
  document.body.appendChild(input);
  input.focus();
  await user.keyboard("{ArrowRight}");
  expect(onNext).toHaveBeenCalledTimes(1);
  input.blur();
  await user.keyboard("{ArrowRight}");
  expect(onNext).toHaveBeenCalledTimes(2);

  rerender({ enabled: false, canPrev: true, canNext: true, onPrev, onNext });
  await user.keyboard("{ArrowLeft}");
  expect(onPrev).toHaveBeenCalledTimes(1);

  rerender({ enabled: true, canPrev: true, canNext: true, onPrev, onNext });
  fireEvent.keyDown(document.body, { key: "ArrowRight", shiftKey: true });
  fireEvent.keyDown(document.body, { key: "ArrowRight", ctrlKey: true });
  expect(onNext).toHaveBeenCalledTimes(2);
  fireEvent.keyDown(document.body, { key: "ArrowRight" });
  expect(onNext).toHaveBeenCalledTimes(3);
});
