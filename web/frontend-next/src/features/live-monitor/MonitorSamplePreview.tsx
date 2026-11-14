import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ImageIcon } from "lucide-react";
import { ResilientImage } from "../../components/ResilientImage";
import { useArrowKeyPaging } from "../../components/useArrowKeyPaging";
import { AssetPagination } from "../training-history/AssetPagination";
import { HistoryImageDialog } from "../training-history/HistoryImageDialog";
import { sampleStep } from "../training-history/historyAssetPresentation";
import {
  fetchResultSampleImages,
  monitorResultKeys,
} from "./api";
import type { HistoryImage } from "../training-history/api";
import "./MonitorSamplePreview.css";

const PAGE_SIZE = 12;

/**
 * Live sample strip for the monitored task. Images are read through the
 * history preview API scoped by task_id, so a finished task keeps its samples
 * and a new task never mixes files. Polling stops as soon as the task is
 * finished; new samples for a running task are picked up by refetchInterval.
 * The strip pages through the fetched window 12 samples at a time, with the
 * same compact pager and arrow-key paging as the history gallery.
 */
export function MonitorSamplePreview({ taskId, active }: { taskId: string; active: boolean }) {
  const [selectedFile, setSelectedFile] = useState<string>();
  const [pageOffset, setPageOffset] = useState(0);
  const query = useQuery({
    queryKey: monitorResultKeys.images(taskId),
    queryFn: ({ signal }) => fetchResultSampleImages(taskId, signal),
    enabled: Boolean(taskId),
    refetchInterval: active ? 5000 : false,
    retry: false,
  });
  const images = useMemo(() => query.data?.images || [], [query.data]);
  // A listing refresh can drop the selected file (or never contain it); keep
  // the dialog honest instead of rendering a stale image reference.
  useEffect(() => {
    if (selectedFile && query.isSuccess && !images.some((image) => image.file === selectedFile)) {
      setSelectedFile(undefined);
    }
  }, [query.isSuccess, selectedFile, images]);
  // A new monitored task (or a shrunken listing) must not leave the strip on
  // a page that no longer exists.
  useEffect(() => { setPageOffset(0); }, [taskId]);
  useEffect(() => {
    if (pageOffset > 0 && pageOffset >= images.length) setPageOffset(0);
  }, [images.length, pageOffset]);
  const nextOffset = pageOffset + PAGE_SIZE < images.length ? pageOffset + PAGE_SIZE : null;
  const visibleImages = images.slice(pageOffset, pageOffset + PAGE_SIZE);
  const selected = images.find((image) => image.file === selectedFile);
  const selectedIndex = selected ? images.indexOf(selected) : -1;
  // Arrow keys page the strip only while the viewer dialog is closed; the
  // dialog handles the same keys to move between individual samples.
  useArrowKeyPaging({
    enabled: selectedIndex < 0,
    canPrev: pageOffset > 0,
    canNext: nextOffset !== null,
    onPrev: () => setPageOffset(Math.max(0, pageOffset - PAGE_SIZE)),
    onNext: () => { if (nextOffset !== null) setPageOffset(nextOffset); },
  });
  return <section className="monitor-sample-panel" aria-label="最新样张">
    <header className="monitor-panel-header">
      <div className="monitor-panel-title">
        <ImageIcon size={16} className="monitor-panel-icon" />
        <h2>最新样张</h2>
      </div>
      {query.data && query.data.total ? <span className="monitor-sample-count">共 {query.data.total} 张</span> : null}
      {images.length > PAGE_SIZE ? <AssetPagination label="样张" offset={pageOffset}
        count={Math.min(PAGE_SIZE, images.length - pageOffset)} total={images.length} size={PAGE_SIZE}
        next={nextOffset} pending={false} compact onChange={setPageOffset} /> : null}
    </header>
    {query.isPending && !query.data ? <p className="monitor-empty" role="status">正在读取样张</p> : null}
    {query.error ? <p className="monitor-empty" role="alert">样张读取失败：{query.error.message}</p> : null}
    {!query.error && images.length === 0 && query.isSuccess ? (
      <p className="monitor-empty">{query.data?.message || "本任务尚未生成样张"}</p>
    ) : null}
    <div className="monitor-sample-strip">
      {visibleImages.map((image) => <SampleCard key={image.file} taskId={taskId} image={image}
        onSelect={setSelectedFile} />)}
    </div>
    {query.data && query.data.total && query.data.total > images.length
      ? <p className="monitor-sample-more">仅读取最新 {images.length} 张，全部样张见历史详情。</p>
      : null}
    {selected && selectedIndex >= 0 ? <HistoryImageDialog image={selected} images={images} index={selectedIndex}
      taskId={taskId} onIndexChange={(index) => setSelectedFile(images[index]?.file)} onClose={() => setSelectedFile(undefined)} /> : null}
  </section>;
}

function SampleCard({ taskId, image, onSelect }: { taskId: string; image: HistoryImage; onSelect: (file: string) => void }) {
  const step = sampleStep(image);
  const label = step !== undefined ? `Step ${step}` : image.name;
  const accessibleLabel = label === image.name ? image.name : `${label} ${image.name}`;
  return <button type="button" className="monitor-sample-card" aria-label={`查看 ${accessibleLabel} 的生成参数`} title={image.name}
    onClick={() => onSelect(image.file)}>
    <ResilientImage className="monitor-sample-thumbnail" src={`/api/preview/image?${new URLSearchParams({ task_id: taskId, file: image.file })}`}
      alt={image.name} loading="lazy" />
    <span className="monitor-sample-step">{label}</span>
  </button>;
}
