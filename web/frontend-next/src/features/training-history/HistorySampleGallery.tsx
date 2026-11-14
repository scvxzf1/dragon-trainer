import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ResilientImage } from "../../components/ResilientImage";
import { useArrowKeyPaging } from "../../components/useArrowKeyPaging";
import { AssetPagination } from "./AssetPagination";
import { HistoryImageDialog } from "./HistoryImageDialog";
import { fetchHistoryImages, historyAssetUrl, type HistoryImage } from "./api";
import { groupHistorySamples, sampleStep } from "./historyAssetPresentation";

const PAGE_SIZE = 60;

export function HistorySampleGallery({ taskId }: { taskId: string }) {
  const [offset, setOffset] = useState(0);
  const [view, setView] = useState("recent");
  const [selectedFile, setSelectedFile] = useState<string>();
  const query = useQuery({
    queryKey: ["history-images", taskId, offset],
    queryFn: ({ signal }) => fetchHistoryImages(taskId, signal, PAGE_SIZE, offset),
  });
  const images = query.data?.images || [];
  const total = query.data?.total ?? 0;
  const groups = view === "grouped" ? groupHistorySamples(images) : [];
  const orderedImages = view === "grouped" ? groups.flatMap((group) => group.images) : images;
  const selectedIndex = orderedImages.findIndex((image) => image.file === selectedFile);
  const nextOffset = query.data?.next_offset === undefined
    ? (offset + PAGE_SIZE < total ? offset + PAGE_SIZE : null)
    : query.data.next_offset;
  const changeOffset = (next: number) => {
    setSelectedFile(undefined);
    setOffset(next);
  };
  useEffect(() => {
    if (selectedFile && query.isSuccess && selectedIndex < 0) setSelectedFile(undefined);
  }, [query.isSuccess, selectedFile, selectedIndex]);
  // Arrow keys flip sample pages only while the viewer dialog is closed; the
  // dialog handles the same keys to move between individual samples.
  useArrowKeyPaging({
    enabled: selectedIndex < 0,
    canPrev: !query.isFetching && offset > 0,
    canNext: !query.isFetching && nextOffset !== null,
    onPrev: () => changeOffset(Math.max(0, offset - PAGE_SIZE)),
    onNext: () => { if (nextOffset !== null) changeOffset(nextOffset); },
  });
  const thumbnail = (image: HistoryImage) => <SampleThumbnail key={image.file} taskId={taskId} image={image} onSelect={setSelectedFile} />;
  const pagination = (compact: boolean) => <AssetPagination label="样张" offset={offset} count={images.length} total={total}
    size={PAGE_SIZE} next={nextOffset} pending={query.isFetching} compact={compact} onChange={changeOffset} />;
  return <section className="history-asset-section" aria-label="训练样张">
    <header className="history-asset-heading">
      <div><h2>训练样张</h2><p>先看效果，再选择权重 · {view === "grouped" ? "本页按 Prompt / Seed 分组，步数从早到晚" : "最新文件优先"} · 方向键 ←/→ 翻页</p></div>
      {total > PAGE_SIZE ? pagination(true) : null}
      <label>浏览方式 <select value={view} onChange={(event) => setView(event.target.value)}>
        <option value="recent">最新样张</option><option value="grouped">Prompt / Seed 演进</option>
      </select></label>
    </header>
    {query.isPending && <p role="status">正在读取样张</p>}
    {query.error && <p role="alert">{query.error.message} <button type="button" onClick={() => query.refetch()}>重试样张</button></p>}
    {view === "grouped" ? groups.map((group) => <section className="history-sample-group" key={group.key}>
      <h3>{group.label}</h3>{group.prompt && <p title={group.prompt}>{group.prompt}</p>}
      <div className="history-image-grid history-sample-evolution">{group.images.map(thumbnail)}</div>
    </section>) : <div className="history-image-grid">{images.map(thumbnail)}</div>}
    {query.data?.images.length === 0 && <p>{query.data.message || "本页暂无样张"}</p>}
    {total > PAGE_SIZE ? pagination(false) : null}
    {selectedIndex >= 0 && <HistoryImageDialog image={orderedImages[selectedIndex]} images={orderedImages} index={selectedIndex}
      taskId={taskId} onIndexChange={(index) => setSelectedFile(orderedImages[index]?.file)} onClose={() => setSelectedFile(undefined)} />}
  </section>;
}

function SampleThumbnail({ taskId, image, onSelect }: { taskId: string; image: HistoryImage; onSelect: (file: string) => void }) {
  const step = sampleStep(image);
  const label = step !== undefined ? `Step ${step}` : image.name;
  const accessibleLabel = label === image.name ? image.name : `${label} ${image.name}`;
  return <button type="button" aria-label={`查看 ${accessibleLabel} 的生成参数`} title={image.name} onClick={() => onSelect(image.file)}>
    <ResilientImage className="history-thumbnail" src={historyAssetUrl(taskId, image.file)} alt={image.name} loading="lazy" />
    <span className="history-sample-step">{label}</span>
  </button>;
}
