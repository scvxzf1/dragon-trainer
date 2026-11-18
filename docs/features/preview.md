# 预览工作区

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/preview`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：
- `web/frontend-next/src/features/preview-workspace/PreviewWorkspace.tsx`
- `web/frontend-next/src/features/training-history/HistoryAssets.tsx`
- `web/routes/preview.py`、`web/services/preview_service.py`、`web/services/image_listing.py`

---

## 1. 来源与筛选范围

通过 Next 主导航的 **预览工作区**（`/next/preview`）集中浏览图片产物与训练权重，支持三种来源：
1. **训练样张**：范围可选 **最新任务**、**指定任务** 或 **配置分组**，支持按最近 7 / 14 / 30 天筛选与分页载入更多；
2. **推理输出**：默认读取 `output/tests`（可在路径设置中修改）；
3. **自定义目录**：读取已保存的项目内或受控绝对图片目录。

在 **训练样张** 与 `/next/history` 历史详情的 **产物** 子页中：
- 点击样张打开左右分栏大图弹窗（左侧支持滚轮/按钮缩放、平移、适应窗口与 100% 重置；右侧显示正负提示词、分辨率、步数、CFG、Seed、采样器及原始元数据，支持复制与下载）；
- 权重列表按轮次/步数展示 `.safetensors` 文件，支持下载或一键将该权重写入当前 `/next/training` 草稿作为热启动来源。

---

## 2. 边界与安全规则

- **有界扫描**：单次列表采用 bounded Top-K 扫描（最多返回 500 张最近图片），仅对进入结果集的图片解析尺寸与 PNG metadata；所选历史任务无样张时明确展示空状态，不静默回退到其他任务的目录。
- **永久删除边界**：删除预览图不可恢复，仅允许删除当前已选受控目录内的直接普通图片文件（单批最多 500 项，拦截子目录、软链、非图片文件与越界路径）。
- **权重下载边界**：仅允许下载受控输出根目录内的权重文件。

---

## 3. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_image_listing.py \
  tests/test_preview_service.py \
  tests/test_cross_domain_delete_boundaries.py \
  tests/test_web_http_contracts.py \
  tests/test_image_test_service.py \
  tests/test_web_route_registry.py \
  -q
```

前端单测与 E2E 位于 `web/frontend-next/src/features/preview-workspace/`，可通过 `.venv/bin/python tasks.py web-next-check` 运行。
