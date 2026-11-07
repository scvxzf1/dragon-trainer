import { NavLink } from "react-router-dom";
import {
  Box,
  Captions,
  Cpu,
  Database,
  History,
  ListTodo,
  Monitor,
  Settings,
  Layers3,
  Sun,
  Moon,
  Menu,
  PanelLeft,
  PanelTop,
  Wrench,
  Image,
  ScanSearch,
  Activity,
  Images,
  FolderTree,
} from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Button } from "../components/ui/Button";

const SIDEBAR_WIDTH_KEY = "dragon-next-ui-v1-sidebar-width";
// 侧边栏宽度按像素持久化，窗口变小或旧值过小时会把导航内容挤出列宽，
// 造成横向滚动与文字被裁切，因此恢复与拖拽都设下限。
const SIDEBAR_MIN_WIDTH = 168;
const NAVIGATION_LAYOUT_KEY = "dragon-next-ui-v1-navigation-layout";

function savedSidebarWidth() {
  try {
    const value = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY));
    if (!Number.isFinite(value) || value <= 0) return null;
    return Math.max(value, SIDEBAR_MIN_WIDTH);
  } catch {
    return null;
  }
}

const LINKS = [
  ["/training", "训练配置", Box],
  ["/datasets", "数据集蓝图", Database],
  ["/data", "数据管理", FolderTree],
  ["/queue", "训练队列", ListTodo],
  ["/monitor", "当前监控", Monitor],
  ["/history", "历史任务", History],
  ["/preview", "预览工作区", Images],
  ["/models", "模型配置", Cpu],
  ["/captioning", "打标工作台", Captions],
  ["/settings", "全局设置", Settings],
] as const;

export function Topbar() {
  const [expanded, setExpanded] = useState(false);
  const [navigationLayout, setNavigationLayout] = useState<"sidebar" | "top">(() => {
    try {
      return localStorage.getItem(NAVIGATION_LAYOUT_KEY) === "top" ? "top" : "sidebar";
    } catch {
      return "sidebar";
    }
  });
  const sidebarWidth = useRef<number | null>(savedSidebarWidth());
  const [widthPercent, setWidthPercent] = useState<number | null>(null);
  useLayoutEffect(() => {
    const updateWidthPercent = () => {
      const layout = document.querySelector<HTMLElement>(".next-layout");
      const sidebar = document.querySelector<HTMLElement>(".app-navigation");
      if (layout && sidebar) {
        setWidthPercent(Math.round((sidebar.getBoundingClientRect().width / layout.getBoundingClientRect().width) * 100));
      }
    };
    if (sidebarWidth.current !== null) {
      document.querySelector<HTMLElement>(".next-layout")?.style.setProperty(
        "--sidebar-width",
        `${sidebarWidth.current}px`,
      );
    }
    updateWidthPercent();
    window.addEventListener("resize", updateWidthPercent);
    return () => window.removeEventListener("resize", updateWidthPercent);
  }, []);

  function resizeSidebar(clientX: number) {
    const layout = document.querySelector<HTMLElement>(".next-layout");
    if (!layout) return;
    const rect = layout.getBoundingClientRect();
    const width = layout.clientWidth;
    const requested = ((clientX - rect.left) / rect.width) * width;
    const maxWidth = Math.max(width - 320, SIDEBAR_MIN_WIDTH);
    const next = Math.round(
      Math.min(Math.max(requested, SIDEBAR_MIN_WIDTH, width * 0.1), maxWidth),
    );
    sidebarWidth.current = next;
    setWidthPercent(Math.round((next / width) * 100));
    layout.style.setProperty("--sidebar-width", `${next}px`);
  }

  function saveSidebarWidth() {
    if (sidebarWidth.current === null) return;
    try {
      localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth.current));
    } catch {
      /* Private browsing can deny storage. */
    }
  }
  function resetSidebarWidth() {
    const layout = document.querySelector<HTMLElement>(".next-layout");
    if (!layout) return;
    sidebarWidth.current = null;
    layout.style.removeProperty("--sidebar-width");
    const sidebar = document.querySelector<HTMLElement>(".app-navigation");
    if (sidebar) {
      setWidthPercent(Math.round((sidebar.getBoundingClientRect().width / layout.getBoundingClientRect().width) * 100));
    }
    try {
      localStorage.removeItem(SIDEBAR_WIDTH_KEY);
    } catch {
      /* Private browsing can deny storage. */
    }
  }
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("dragon-next-ui-v1-theme") === "light"
        ? "light"
        : "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("dragon-next-ui-v1-theme", theme);
    } catch {
      /* Private browsing can deny storage. */
    }
  }, [theme]);
  useLayoutEffect(() => {
    document.querySelector<HTMLElement>(".next-layout")?.setAttribute("data-navigation-layout", navigationLayout);
    try {
      localStorage.setItem(NAVIGATION_LAYOUT_KEY, navigationLayout);
    } catch {
      /* Private browsing can deny storage. */
    }
  }, [navigationLayout]);
  return (
    <aside className="app-navigation" data-expanded={expanded}>
      <div
        className="app-navigation-resizer"
        role="separator"
        aria-label="调整侧边栏宽度"
        aria-orientation="vertical"
        aria-valuemin={10}
        aria-valuemax={100}
        aria-valuenow={widthPercent ?? undefined}
        tabIndex={0}
        title="拖动调整宽度，双击恢复默认"
        onDoubleClick={resetSidebarWidth}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            resizeSidebar(event.clientX);
          }
        }}
        onPointerUp={saveSidebarWidth}
        onPointerCancel={saveSidebarWidth}
        onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const layout = document.querySelector<HTMLElement>(".next-layout");
          if (!layout) return;
          const rect = layout.getBoundingClientRect();
          const current = (document.querySelector<HTMLElement>(".app-navigation")?.getBoundingClientRect().width ?? rect.width * 0.16) / rect.width * layout.clientWidth;
          const target = event.key === "Home"
            ? layout.clientWidth * 0.1
            : event.key === "End"
              ? layout.clientWidth - 320
              : current + (event.key === "ArrowRight" ? 10 : -10);
          resizeSidebar(rect.left + (target / layout.clientWidth) * rect.width);
          saveSidebarWidth();
        }}
      />
      <div className="app-brand">
        <Layers3 size={23} />
        <strong>
          Dragon<span>TRAINER WORKSPACE</span>
        </strong>
        <Button
          variant="ghost"
          size="icon"
          className="nav-toggle"
          aria-label="切换导航"
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          <Menu size={19} />
        </Button>
      </div>
      <p className="nav-section-label">工作空间</p>
      <nav aria-label="主导航" className="app-links">
        {LINKS.map(([to, label, Icon]) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/training" || to === "/datasets" || to === "/data"}
            aria-label={label}
            title={label}
            onClick={() => setExpanded(false)}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="nav-footer">
        <details>
          <summary aria-label="辅助工具" title="辅助工具"><Wrench size={16} aria-hidden="true" /><span>辅助工具</span></summary>
          <div className="nav-aux-links">
            <NavLink to="/image-test" aria-label="生图测试" title="生图测试" onClick={() => setExpanded(false)}><Image size={16} aria-hidden="true" /><span>生图测试</span></NavLink>
            <NavLink to="/weight-analysis" aria-label="权重分析" title="权重分析" onClick={() => setExpanded(false)}><ScanSearch size={16} aria-hidden="true" /><span>权重分析</span></NavLink>
            <NavLink to="/environment" aria-label="环境检测" title="环境检测" onClick={() => setExpanded(false)}><Activity size={16} aria-hidden="true" /><span>环境检测</span></NavLink>
          </div>
        </details>
        <div className="nav-appearance-controls">
          <Button
            variant="ghost"
            aria-label={theme === "dark" ? "浅色外观" : "深色外观"}
            title={theme === "dark" ? "浅色外观" : "深色外观"}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            {theme === "dark" ? "浅色外观" : "深色外观"}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label={navigationLayout === "sidebar" ? "切换为顶部栏" : "切换为侧边栏"}
            title={navigationLayout === "sidebar" ? "切换为顶部栏" : "切换为侧边栏"}
            onClick={() => { setExpanded(false); setNavigationLayout(navigationLayout === "sidebar" ? "top" : "sidebar"); }}
          >
            {navigationLayout === "sidebar" ? <PanelTop size={17} /> : <PanelLeft size={17} />}
          </Button>
        </div>
      </div>
    </aside>
  );
}
