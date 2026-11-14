import { ChevronLeft, ChevronRight } from "lucide-react";

export function AssetPagination({ label, offset, count, total, size, next, pending, onChange, compact }: {
  label: string; offset: number; count: number; total: number; size: number;
  next?: number | null; pending: boolean; onChange: (offset: number) => void; compact?: boolean;
}) {
  const nextOffset = next === undefined ? (offset + size < total ? offset + size : null) : next;
  return <nav className="toolbar" aria-label={`${label}分页`}>
    {!compact && <span>{count ? `${offset + 1}-${offset + count}` : "0"} / {total}</span>}
    <button type="button" aria-label={`上一页${label}`} title={`上一页${label}`} disabled={pending || offset === 0} onClick={() => onChange(Math.max(0, offset - size))}><ChevronLeft size={16} /></button>
    <button type="button" aria-label={`下一页${label}`} title={`下一页${label}`} disabled={pending || nextOffset === null} onClick={() => nextOffset !== null && onChange(nextOffset)}><ChevronRight size={16} /></button>
  </nav>;
}
