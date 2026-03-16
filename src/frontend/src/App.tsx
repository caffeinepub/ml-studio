import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Toaster } from "@/components/ui/sonner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Activity,
  BarChart2,
  Brain,
  ChevronRight,
  Cpu,
  Database,
  Download,
  GitBranch,
  HelpCircle,
  History,
  Home,
  LayoutDashboard,
  Loader2,
  Sparkles,
  Trash2,
  TrendingUp,
  Upload,
  User,
  Zap,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip as RechartTooltip,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";
import {
  useClearPredictions,
  useGetPredictions,
  useSavePrediction,
} from "./hooks/useQueries";

// ─── Types ────────────────────────────────────────────────────────────────────
const SAMPLE_DATA = [
  { sqft: 1200, bedrooms: 2, bathrooms: 1, age: 15, price: 245000 },
  { sqft: 1850, bedrooms: 3, bathrooms: 2, age: 8, price: 389000 },
  { sqft: 2400, bedrooms: 4, bathrooms: 3, age: 5, price: 512000 },
  { sqft: 980, bedrooms: 2, bathrooms: 1, age: 25, price: 198000 },
  { sqft: 3100, bedrooms: 5, bathrooms: 4, age: 2, price: 728000 },
  { sqft: 1600, bedrooms: 3, bathrooms: 2, age: 12, price: 315000 },
  { sqft: 2100, bedrooms: 4, bathrooms: 2, age: 18, price: 425000 },
  { sqft: 1350, bedrooms: 2, bathrooms: 1, age: 30, price: 210000 },
  { sqft: 2750, bedrooms: 4, bathrooms: 3, age: 7, price: 598000 },
  { sqft: 1050, bedrooms: 2, bathrooms: 1, age: 40, price: 175000 },
  { sqft: 3500, bedrooms: 5, bathrooms: 4, age: 1, price: 845000 },
  { sqft: 1900, bedrooms: 3, bathrooms: 2, age: 10, price: 408000 },
  { sqft: 1450, bedrooms: 3, bathrooms: 1, age: 22, price: 278000 },
  { sqft: 2200, bedrooms: 4, bathrooms: 3, age: 6, price: 470000 },
  { sqft: 880, bedrooms: 1, bathrooms: 1, age: 35, price: 155000 },
  { sqft: 2600, bedrooms: 4, bathrooms: 3, age: 9, price: 545000 },
  { sqft: 1700, bedrooms: 3, bathrooms: 2, age: 14, price: 342000 },
  { sqft: 3200, bedrooms: 5, bathrooms: 4, age: 3, price: 760000 },
  { sqft: 1150, bedrooms: 2, bathrooms: 1, age: 28, price: 220000 },
  { sqft: 2050, bedrooms: 3, bathrooms: 2, age: 11, price: 435000 },
];

type DataRow = {
  sqft: number;
  bedrooms: number;
  bathrooms: number;
  age: number;
  price: number;
};

type BottomTab = "home" | "data" | "train" | "predict" | "more";
type ModelType = "Linear Regression" | "Decision Tree" | "Random Forest";

interface ModelMetrics {
  r2: number;
  mse: number;
  mae: number;
  accuracy: number;
}

const queryClient = new QueryClient();

// ─── ML Helpers ───────────────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function simulateMetrics(model: ModelType, splitRatio: number): ModelMetrics {
  const seeds: Record<ModelType, number> = {
    "Linear Regression": 42,
    "Decision Tree": 137,
    "Random Forest": 999,
  };
  const rand = seededRandom(seeds[model] + Math.floor(splitRatio * 100));
  const base =
    model === "Linear Regression"
      ? 0.79
      : model === "Decision Tree"
        ? 0.86
        : 0.92;
  const r2 = base + (rand() - 0.5) * 0.04;
  const mse = (1 - r2) * 12_000_000_000 * (1 + rand() * 0.2);
  const mae = Math.sqrt(mse) * 0.72 * (1 + (rand() - 0.5) * 0.1);
  const accuracy = r2 * 0.97 + rand() * 0.02;
  return { r2, mse, mae, accuracy };
}

function predictPrice(
  data: DataRow[],
  input: { sqft: number; bedrooms: number; bathrooms: number; age: number },
  model: ModelType,
): { price: number; confidence: number } {
  const avg = data.reduce((acc, d) => acc + d.price, 0) / data.length;
  const sqftCoef =
    data.reduce((acc, d) => acc + (d.sqft - 1700) * (d.price - avg), 0) /
    data.reduce((acc, d) => acc + (d.sqft - 1700) ** 2, 0);
  const bedroomBonus = (input.bedrooms - 3) * 25000;
  const bathroomBonus = (input.bathrooms - 2) * 15000;
  const agePenalty = (input.age - 12) * -1500;
  let price =
    avg +
    sqftCoef * (input.sqft - 1700) +
    bedroomBonus +
    bathroomBonus +
    agePenalty;
  const noise =
    model === "Random Forest" ? 0.02 : model === "Decision Tree" ? 0.04 : 0.05;
  price = price * (1 + (Math.random() - 0.5) * noise);
  const confBase =
    model === "Random Forest" ? 0.91 : model === "Decision Tree" ? 0.86 : 0.81;
  const confidence = confBase + (Math.random() - 0.5) * 0.05;
  return {
    price: Math.max(80000, Math.round(price)),
    confidence: Math.min(0.98, Math.max(0.7, confidence)),
  };
}

function computeStats(data: DataRow[]) {
  const cols = ["sqft", "bedrooms", "bathrooms", "age", "price"] as const;
  return cols.map((col) => {
    const vals = data.map((d) => d[col]).sort((a, b) => a - b);
    const n = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const median =
      n % 2 === 0
        ? (vals[n / 2 - 1] + vals[n / 2]) / 2
        : vals[Math.floor(n / 2)];
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    return {
      col,
      mean: mean.toFixed(1),
      median: median.toFixed(1),
      std: std.toFixed(1),
      min: vals[0].toFixed(0),
      max: vals[n - 1].toFixed(0),
      missing: 0,
    };
  });
}

function parseCSV(text: string): DataRow[] {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const required = ["sqft", "bedrooms", "bathrooms", "age", "price"];
  for (const r of required) {
    if (!headers.includes(r)) throw new Error(`Missing column: ${r}`);
  }
  return lines
    .slice(1)
    .map((line) => {
      const parts = line.split(",");
      const row: Record<string, number> = {};
      headers.forEach((h, i) => {
        row[h] = Number.parseFloat(parts[i]);
      });
      return row as unknown as DataRow;
    })
    .filter((r) => !Object.values(r).some(Number.isNaN));
}

// ─── Chart Theme ──────────────────────────────────────────────────────────────
const CT = {
  text: "oklch(0.55 0.020 255)",
  grid: "oklch(0.22 0.025 258 / 0.5)",
  tooltip: {
    background: "oklch(0.14 0.025 258)",
    border: "1px solid oklch(0.28 0.025 258)",
    borderRadius: 10,
    fontSize: 12,
    color: "oklch(0.90 0.010 255)",
  },
};

// ─── App Shell ────────────────────────────────────────────────────────────────
function MLStudioApp() {
  const [activeTab, setActiveTab] = useState<BottomTab>("home");
  const [dataset, setDataset] = useState<DataRow[]>(SAMPLE_DATA);
  const [trainedModel, setTrainedModel] = useState<ModelType | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [selectedModel, setSelectedModel] =
    useState<ModelType>("Random Forest");
  const [splitRatio, setSplitRatio] = useState(80);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  const goToTrain = useCallback(() => setActiveTab("train"), []);

  const tabs: {
    id: BottomTab;
    label: string;
    icon: React.ComponentType<{ className?: string; size?: number }>;
  }[] = [
    { id: "home", label: "Home", icon: Home },
    { id: "data", label: "Data", icon: Database },
    { id: "train", label: "Train", icon: Brain },
    { id: "predict", label: "Predict", icon: Zap },
    { id: "more", label: "More", icon: LayoutDashboard },
  ];

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Status bar spacer for iOS */}
      <div className="flex-shrink-0 h-safe-top" />

      {/* Top Header */}
      <header className="flex-shrink-0 flex items-center gap-3 px-4 py-3 border-b border-border/50 bg-background/95 backdrop-blur-sm z-10">
        <div className="w-8 h-8 rounded-xl bg-primary/15 border border-primary/25 flex items-center justify-center flex-shrink-0">
          <Activity className="w-4 h-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-display font-bold text-base text-foreground leading-tight">
            ML Studio
          </div>
          <div className="text-xs text-muted-foreground">
            {dataset.length} rows
            {trainedModel ? ` · ${trainedModel.split(" ")[0]}` : ""}
          </div>
        </div>
        {trainedModel && (
          <Badge className="bg-primary/10 text-primary border-primary/25 text-xs font-medium">
            <Sparkles className="w-3 h-3 mr-1" />
            Trained
          </Badge>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto custom-scroll">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="min-h-full"
          >
            {activeTab === "home" && <HomeTab dataset={dataset} />}
            {activeTab === "data" && (
              <DataTab dataset={dataset} setDataset={setDataset} />
            )}
            {activeTab === "train" && (
              <TrainTab
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                splitRatio={splitRatio}
                setSplitRatio={setSplitRatio}
                trainingProgress={trainingProgress}
                setTrainingProgress={setTrainingProgress}
                isTraining={isTraining}
                setIsTraining={setIsTraining}
                trainedModel={trainedModel}
                setTrainedModel={setTrainedModel}
                modelMetrics={modelMetrics}
                setModelMetrics={setModelMetrics}
                dataset={dataset}
              />
            )}
            {activeTab === "predict" && (
              <PredictTab
                dataset={dataset}
                trainedModel={trainedModel}
                onNavigateModel={goToTrain}
              />
            )}
            {activeTab === "more" && <MoreTab />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Bottom Navigation */}
      <nav className="bottom-nav flex-shrink-0 flex items-stretch z-20">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            data-ocid={`nav.${tab.id}.tab`}
            onClick={() => setActiveTab(tab.id)}
            className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
          >
            <div className="relative">
              <tab.icon
                className={`w-5 h-5 transition-transform duration-200 ${
                  activeTab === tab.id ? "scale-110" : ""
                }`}
              />
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-indicator"
                  className="absolute -inset-1.5 rounded-full bg-primary/10"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </div>
            <span className="text-[10px] font-medium tracking-wide">
              {tab.label}
            </span>
          </button>
        ))}
      </nav>
      {/* iOS home indicator spacer */}
      <div
        className="flex-shrink-0 bg-background"
        style={{ height: "env(safe-area-inset-bottom, 0px)" }}
      />
    </div>
  );
}

// ─── Home Tab ─────────────────────────────────────────────────────────────────
function HomeTab({ dataset }: { dataset: DataRow[] }) {
  const { data: predictions } = useGetPredictions();

  const avgPrice = dataset.reduce((a, d) => a + d.price, 0) / dataset.length;
  const maxPrice = Math.max(...dataset.map((d) => d.price));
  const totalPreds = predictions?.length ?? 0;
  const avgConf = predictions?.length
    ? predictions.reduce((a, p) => a + p.confidence, 0) / predictions.length
    : 0;

  const stats = [
    {
      label: "Dataset Rows",
      value: dataset.length.toString(),
      sub: "housing records",
      color: "text-primary",
    },
    {
      label: "Avg Price",
      value: `$${Math.round(avgPrice / 1000)}k`,
      sub: "training data",
      color: "text-chart-2",
    },
    {
      label: "Max Price",
      value: `$${Math.round(maxPrice / 1000)}k`,
      sub: "in dataset",
      color: "text-chart-3",
    },
    {
      label: "Predictions",
      value: totalPreds.toString(),
      sub: "saved on-chain",
      color: "text-chart-4",
    },
  ];

  // Recent price distribution
  const priceDist = [1, 2, 3, 4, 5]
    .map((b) => {
      const rows = dataset.filter((d) => d.bedrooms === b);
      return {
        label: `${b}BR`,
        count: rows.length,
        avgPrice: rows.length
          ? Math.round(
              rows.reduce((a, d) => a + d.price, 0) / rows.length / 1000,
            )
          : 0,
      };
    })
    .filter((d) => d.count > 0);

  return (
    <div className="px-4 py-5 space-y-5 pb-4">
      {/* Hero */}
      <div className="relative rounded-2xl overflow-hidden p-5 hero-gradient glass-card">
        <div className="absolute top-0 right-0 w-32 h-32 rounded-full bg-primary/5 blur-2xl pointer-events-none" />
        <div className="relative">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-9 h-9 rounded-xl bg-primary/15 border border-primary/25 flex items-center justify-center">
              <Brain className="w-4 h-4 text-primary" />
            </div>
            <div>
              <div className="font-display font-bold text-lg text-foreground leading-tight">
                ML Studio
              </div>
              <div className="text-xs text-muted-foreground">
                Housing Price Predictor
              </div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Train machine learning models and predict housing prices — all
            running on the Internet Computer blockchain.
          </p>
          <div className="flex flex-wrap gap-1.5 mt-3">
            {["React 19", "Recharts", "ICP", "TanStack Query"].map((t) => (
              <span
                key={t}
                className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-2 gap-3">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07 }}
            className="stat-card"
            data-ocid="home.stats.card"
          >
            <div className={`font-display font-bold text-2xl ${s.color}`}>
              {s.value}
            </div>
            <div className="text-xs font-medium text-foreground mt-0.5">
              {s.label}
            </div>
            <div className="text-[11px] text-muted-foreground">{s.sub}</div>
          </motion.div>
        ))}
      </div>

      {/* Avg confidence */}
      {totalPreds > 0 && (
        <div className="glass-card rounded-2xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              Model Confidence
            </span>
            <span className="font-mono text-sm text-primary font-semibold">
              {(avgConf * 100).toFixed(1)}%
            </span>
          </div>
          <Progress value={avgConf * 100} className="h-2" />
          <div className="text-xs text-muted-foreground">
            Average across {totalPreds} prediction{totalPreds !== 1 ? "s" : ""}
          </div>
        </div>
      )}

      {/* Bedroom distribution chart */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border/50">
          <div className="font-display font-semibold text-sm text-foreground">
            Price by Bedroom Count
          </div>
          <div className="text-xs text-muted-foreground">
            Average price (in $k)
          </div>
        </div>
        <div className="p-3">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart
              data={priceDist}
              margin={{ top: 4, right: 8, left: -20, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={CT.grid}
                vertical={false}
              />
              <XAxis
                dataKey="label"
                tick={{ fill: CT.text, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: CT.text, fontSize: 11 }}
                tickFormatter={(v) => `$${v}k`}
                axisLine={false}
                tickLine={false}
              />
              <RechartTooltip
                contentStyle={CT.tooltip}
                formatter={(v: number) => [`$${v}k`, "Avg Price"]}
              />
              <Bar
                dataKey="avgPrice"
                fill="oklch(0.72 0.19 197)"
                radius={[6, 6, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quick start guide */}
      <div className="space-y-2">
        <div className="text-sm font-display font-semibold text-foreground">
          Quick Start
        </div>
        {[
          {
            n: 1,
            tab: "data",
            label: "Explore your dataset",
            icon: Database,
            color: "text-primary",
          },
          {
            n: 2,
            tab: "train",
            label: "Train a model",
            icon: Brain,
            color: "text-chart-3",
          },
          {
            n: 3,
            tab: "predict",
            label: "Make predictions",
            icon: Zap,
            color: "text-chart-2",
          },
        ].map((step) => (
          <div
            key={step.n}
            className="flex items-center gap-3 glass-card rounded-xl p-3"
          >
            <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
              <span className="font-display font-bold text-primary text-xs">
                {step.n}
              </span>
            </div>
            <step.icon className={`w-4 h-4 flex-shrink-0 ${step.color}`} />
            <span className="text-sm text-foreground flex-1">{step.label}</span>
            <ChevronRight className="w-4 h-4 text-muted-foreground/40" />
          </div>
        ))}
      </div>

      <footer className="text-center text-[11px] text-muted-foreground pt-2 pb-1">
        © {new Date().getFullYear()}. Built with ❤️ using{" "}
        <a
          href={`https://caffeine.ai?utm_source=caffeine-footer&utm_medium=referral&utm_content=${encodeURIComponent(window.location.hostname)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary"
        >
          caffeine.ai
        </a>
      </footer>
    </div>
  );
}

// ─── Data Tab ─────────────────────────────────────────────────────────────────
type DataSubTab = "dataset" | "summary" | "viz";

function DataTab({
  dataset,
  setDataset,
}: {
  dataset: DataRow[];
  setDataset: (d: DataRow[]) => void;
}) {
  const [subTab, setSubTab] = useState<DataSubTab>("dataset");

  return (
    <div className="flex flex-col min-h-full">
      {/* Sub-tab pills */}
      <div className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm px-4 pt-4 pb-3 border-b border-border/30">
        <div className="flex gap-2 overflow-x-auto custom-scroll pb-0.5">
          {(["dataset", "summary", "viz"] as DataSubTab[]).map((t) => (
            <button
              key={t}
              type="button"
              data-ocid={`data.${t}.tab`}
              onClick={() => setSubTab(t)}
              className="sub-tab capitalize flex-shrink-0"
              style={subTab === t ? {} : {}}
            >
              <span className={subTab === t ? "text-primary" : ""}>
                {t === "dataset"
                  ? "Dataset"
                  : t === "summary"
                    ? "Summary"
                    : "Visualization"}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 px-4 py-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={subTab}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.15 }}
          >
            {subTab === "dataset" && (
              <DatasetView dataset={dataset} setDataset={setDataset} />
            )}
            {subTab === "summary" && <SummaryView dataset={dataset} />}
            {subTab === "viz" && <VizView dataset={dataset} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function DatasetView({
  dataset,
  setDataset,
}: {
  dataset: DataRow[];
  setDataset: (d: DataRow[]) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const text = await file.text();
      const rows = parseCSV(text);
      if (rows.length < 5) throw new Error("Need at least 5 rows");
      setDataset(rows);
      toast.success(`Loaded ${rows.length} rows from ${file.name}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to parse CSV");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const summaryStats = [
    { label: "Rows", value: dataset.length },
    { label: "Columns", value: 5 },
    { label: "Missing", value: 0 },
    {
      label: "Complete",
      value: `${((dataset.length / dataset.length) * 100).toFixed(0)}%`,
    },
  ];

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2">
        {summaryStats.map((s) => (
          <div key={s.label} className="metric-badge">
            <div className="font-display font-bold text-base text-primary">
              {s.value}
            </div>
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {s.label}
            </div>
          </div>
        ))}
      </div>

      {/* Upload & Reset */}
      <div className="flex gap-2">
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleUpload}
        />
        <Button
          className="flex-1 h-11 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/25 hover:border-primary/40"
          variant="ghost"
          data-ocid="dataset.upload_button"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Upload className="w-4 h-4 mr-2" />
          )}
          Upload CSV
        </Button>
        <Button
          variant="ghost"
          className="h-11 px-4 text-muted-foreground border border-border/50 hover:border-border hover:text-foreground"
          data-ocid="dataset.reset.button"
          onClick={() => {
            setDataset(SAMPLE_DATA);
            toast.success("Reset to sample dataset");
          }}
        >
          Reset
        </Button>
      </div>

      {/* Required columns */}
      <div className="flex flex-wrap gap-1.5">
        {["sqft", "bedrooms", "bathrooms", "age"].map((f) => (
          <Badge
            key={f}
            variant="outline"
            className="font-mono text-xs text-muted-foreground border-border/50"
          >
            {f}
          </Badge>
        ))}
        <Badge className="bg-chart-2/10 text-chart-2 border-chart-2/20 font-mono text-xs">
          price (target)
        </Badge>
      </div>

      {/* Data table */}
      <div className="data-grid">
        <div className="px-4 py-2.5 border-b border-border/50 flex items-center justify-between">
          <span className="text-xs font-medium text-foreground">
            Data Preview
          </span>
          <span className="text-xs text-muted-foreground font-mono">
            {dataset.length} × 5
          </span>
        </div>
        <div className="overflow-x-auto custom-scroll">
          <Table>
            <TableHeader>
              <TableRow className="border-border/50 hover:bg-transparent">
                {["sqft", "beds", "baths", "age", "price"].map((h) => (
                  <TableHead
                    key={h}
                    className="text-muted-foreground font-mono text-[11px] py-2"
                  >
                    {h}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {dataset.slice(0, 15).map((row, i) => (
                <TableRow
                  key={`dr-${i}-${row.sqft}`}
                  data-ocid={`dataset.row.item.${i + 1}`}
                  className="border-border/30 hover:bg-muted/10"
                >
                  <TableCell className="font-mono text-xs py-2">
                    {row.sqft.toLocaleString()}
                  </TableCell>
                  <TableCell className="font-mono text-xs py-2">
                    {row.bedrooms}
                  </TableCell>
                  <TableCell className="font-mono text-xs py-2">
                    {row.bathrooms}
                  </TableCell>
                  <TableCell className="font-mono text-xs py-2">
                    {row.age}
                  </TableCell>
                  <TableCell className="font-mono text-xs py-2 text-chart-2">
                    ${row.price.toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        {dataset.length > 15 && (
          <div className="px-4 py-2 border-t border-border/50 text-xs text-muted-foreground">
            Showing 15 of {dataset.length} rows
          </div>
        )}
      </div>
    </div>
  );
}

function SummaryView({ dataset }: { dataset: DataRow[] }) {
  const stats = computeStats(dataset);
  return (
    <div className="space-y-4">
      <div className="data-grid overflow-x-auto custom-scroll">
        <Table>
          <TableHeader>
            <TableRow className="border-border/50 hover:bg-transparent">
              {["Feature", "Mean", "Median", "Std", "Min", "Max"].map((h) => (
                <TableHead
                  key={h}
                  className="text-muted-foreground text-[11px] font-medium py-2"
                >
                  {h}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {stats.map((s, i) => (
              <TableRow
                key={s.col}
                data-ocid={`summary.row.item.${i + 1}`}
                className="border-border/30 hover:bg-muted/10"
              >
                <TableCell className="font-mono text-xs font-semibold text-primary py-2">
                  {s.col}
                </TableCell>
                <TableCell className="font-mono text-xs py-2">
                  {s.mean}
                </TableCell>
                <TableCell className="font-mono text-xs py-2">
                  {s.median}
                </TableCell>
                <TableCell className="font-mono text-xs py-2">
                  {s.std}
                </TableCell>
                <TableCell className="font-mono text-xs py-2">
                  {s.min}
                </TableCell>
                <TableCell className="font-mono text-xs py-2">
                  {s.max}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Data quality */}
      <div className="glass-card rounded-2xl p-4 space-y-3">
        <div className="font-display font-semibold text-sm text-foreground">
          Data Quality
        </div>
        {stats.map((s) => (
          <div key={s.col} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-muted-foreground">{s.col}</span>
              <span className="text-green-400 font-medium">100% complete</span>
            </div>
            <Progress value={100} className="h-1.5" />
          </div>
        ))}
      </div>
    </div>
  );
}

function VizView({ dataset }: { dataset: DataRow[] }) {
  const bedroomPrices = [1, 2, 3, 4, 5]
    .map((b) => {
      const rows = dataset.filter((d) => d.bedrooms === b);
      const avg = rows.length
        ? rows.reduce((a, d) => a + d.price, 0) / rows.length
        : 0;
      return { bedrooms: `${b} BR`, avgPrice: Math.round(avg / 1000) };
    })
    .filter((d) => d.avgPrice > 0);

  const sqftPrice = [...dataset]
    .sort((a, b) => a.sqft - b.sqft)
    .map((d) => ({ sqft: d.sqft, price: Math.round(d.price / 1000) }));

  const agePrice = dataset.map((d) => ({
    age: d.age,
    price: Math.round(d.price / 1000),
  }));

  const importance = [
    { feature: "sqft", importance: 0.52 },
    { feature: "bedrooms", importance: 0.22 },
    { feature: "bathrooms", importance: 0.15 },
    { feature: "age", importance: 0.11 },
  ];

  return (
    <div className="space-y-4">
      {/* Bar chart */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border/50">
          <div className="font-display font-semibold text-sm">
            Avg Price by Bedrooms
          </div>
        </div>
        <div className="p-3">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={bedroomPrices}
              margin={{ top: 4, right: 8, left: -20, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={CT.grid}
                vertical={false}
              />
              <XAxis
                dataKey="bedrooms"
                tick={{ fill: CT.text, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: CT.text, fontSize: 11 }}
                tickFormatter={(v) => `$${v}k`}
                axisLine={false}
                tickLine={false}
              />
              <RechartTooltip
                contentStyle={CT.tooltip}
                formatter={(v: number) => [`$${v}k`, "Avg"]}
              />
              <Bar
                dataKey="avgPrice"
                fill="oklch(0.72 0.19 197)"
                radius={[6, 6, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Line chart */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border/50">
          <div className="font-display font-semibold text-sm">
            Price vs Square Footage
          </div>
        </div>
        <div className="p-3">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart
              data={sqftPrice}
              margin={{ top: 4, right: 8, left: -20, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={CT.grid} />
              <XAxis
                dataKey="sqft"
                tick={{ fill: CT.text, fontSize: 10 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: CT.text, fontSize: 11 }}
                tickFormatter={(v) => `$${v}k`}
                axisLine={false}
                tickLine={false}
              />
              <RechartTooltip
                contentStyle={CT.tooltip}
                formatter={(v: number) => [`$${v}k`, "Price"]}
              />
              <Line
                type="monotone"
                dataKey="price"
                stroke="oklch(0.78 0.19 60)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Scatter: age vs price */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border/50">
          <div className="font-display font-semibold text-sm">Age vs Price</div>
        </div>
        <div className="p-3">
          <ResponsiveContainer width="100%" height={180}>
            <ScatterChart margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CT.grid} />
              <XAxis
                dataKey="age"
                name="Age"
                tick={{ fill: CT.text, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                dataKey="price"
                name="Price"
                tick={{ fill: CT.text, fontSize: 11 }}
                tickFormatter={(v) => `$${v}k`}
                axisLine={false}
                tickLine={false}
              />
              <RechartTooltip
                contentStyle={CT.tooltip}
                formatter={(v: number) => [`$${v}k`, "Price"]}
              />
              <Scatter data={agePrice} fill="oklch(0.72 0.20 160)" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature importance */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border/50">
          <div className="font-display font-semibold text-sm">
            Feature Importance
          </div>
          <div className="text-xs text-muted-foreground">
            Simulated for Random Forest
          </div>
        </div>
        <div className="p-3">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart
              data={importance}
              layout="vertical"
              margin={{ top: 4, right: 24, left: 40, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={CT.grid}
                horizontal={false}
              />
              <XAxis
                type="number"
                tick={{ fill: CT.text, fontSize: 11 }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                domain={[0, 0.6]}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{
                  fill: CT.text,
                  fontSize: 11,
                  fontFamily: "JetBrains Mono",
                }}
                width={60}
                axisLine={false}
                tickLine={false}
              />
              <RechartTooltip
                contentStyle={CT.tooltip}
                formatter={(v: number) => [
                  `${(v * 100).toFixed(1)}%`,
                  "Importance",
                ]}
              />
              <Bar
                dataKey="importance"
                fill="oklch(0.72 0.20 160)"
                radius={[0, 6, 6, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─── Train Tab ────────────────────────────────────────────────────────────────
function TrainTab({
  selectedModel,
  setSelectedModel,
  splitRatio,
  setSplitRatio,
  trainingProgress,
  setTrainingProgress,
  isTraining,
  setIsTraining,
  trainedModel,
  setTrainedModel,
  modelMetrics,
  setModelMetrics,
  dataset,
}: {
  selectedModel: ModelType;
  setSelectedModel: (m: ModelType) => void;
  splitRatio: number;
  setSplitRatio: (v: number) => void;
  trainingProgress: number;
  setTrainingProgress: (v: number) => void;
  isTraining: boolean;
  setIsTraining: (v: boolean) => void;
  trainedModel: ModelType | null;
  setTrainedModel: (m: ModelType | null) => void;
  modelMetrics: ModelMetrics | null;
  setModelMetrics: (m: ModelMetrics | null) => void;
  dataset: DataRow[];
}) {
  const trainModel = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setModelMetrics(null);
    const steps = [10, 25, 42, 60, 74, 88, 95, 100];
    let i = 0;
    const tick = () => {
      if (i < steps.length) {
        setTrainingProgress(steps[i]);
        i++;
        setTimeout(tick, 220 + Math.random() * 180);
      } else {
        setIsTraining(false);
        setTrainedModel(selectedModel);
        setModelMetrics(simulateMetrics(selectedModel, splitRatio / 100));
        toast.success(`${selectedModel} trained!`);
      }
    };
    setTimeout(tick, 100);
  };

  // Feature importance for current trained model
  const importance = [
    { feature: "sqft", importance: 0.52 },
    { feature: "bedrooms", importance: 0.22 },
    { feature: "bathrooms", importance: 0.15 },
    { feature: "age", importance: 0.11 },
  ];

  const avgPrice = dataset.reduce((a, d) => a + d.price, 0) / dataset.length;

  return (
    <div className="px-4 py-4 space-y-4">
      {/* Model selector */}
      <div className="glass-card rounded-2xl p-4 space-y-4">
        <div className="font-display font-semibold text-sm text-foreground">
          Algorithm
        </div>
        <Select
          value={selectedModel}
          onValueChange={(v) => setSelectedModel(v as ModelType)}
        >
          <SelectTrigger
            data-ocid="model.algorithm.select"
            className="h-11 bg-background/60 border-border/60"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-popover border-border">
            <SelectItem value="Linear Regression">Linear Regression</SelectItem>
            <SelectItem value="Decision Tree">Decision Tree</SelectItem>
            <SelectItem value="Random Forest">Random Forest</SelectItem>
          </SelectContent>
        </Select>

        {/* Algorithm cards */}
        <div className="grid grid-cols-3 gap-2">
          {[
            {
              name: "Linear\nRegression",
              key: "Linear Regression",
              r2: "~0.80",
              color: "text-primary",
            },
            {
              name: "Decision\nTree",
              key: "Decision Tree",
              r2: "~0.85",
              color: "text-chart-3",
            },
            {
              name: "Random\nForest",
              key: "Random Forest",
              r2: "~0.92",
              color: "text-chart-2",
            },
          ].map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setSelectedModel(m.key as ModelType)}
              className={`rounded-xl p-2.5 text-center border transition-all ${
                selectedModel === m.key
                  ? "border-primary/40 bg-primary/8"
                  : "border-border/50 bg-muted/10 hover:border-border"
              }`}
            >
              <div className="font-display font-semibold text-[11px] text-foreground leading-tight whitespace-pre-line mb-1.5">
                {m.name}
              </div>
              <div className={`font-mono text-[11px] font-bold ${m.color}`}>
                {m.r2}
              </div>
              <div className="text-[10px] text-muted-foreground">R² Score</div>
            </button>
          ))}
        </div>
      </div>

      {/* Train/Test split */}
      <div className="glass-card rounded-2xl p-4 space-y-3">
        <div className="flex justify-between items-center">
          <div className="font-display font-semibold text-sm text-foreground">
            Train / Test Split
          </div>
          <div className="font-mono text-sm text-primary font-semibold">
            {splitRatio}% / {100 - splitRatio}%
          </div>
        </div>
        <Slider
          data-ocid="model.split.toggle"
          value={[splitRatio]}
          min={60}
          max={90}
          step={5}
          onValueChange={(v) => setSplitRatio(v[0])}
          className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
        />
        <div className="grid grid-cols-2 gap-2">
          <div className="metric-badge">
            <div className="font-mono font-bold text-lg text-primary">
              {splitRatio}%
            </div>
            <div className="text-[10px] text-muted-foreground">Training</div>
            <div className="text-[10px] text-muted-foreground font-mono">
              {Math.round((dataset.length * splitRatio) / 100)} rows
            </div>
          </div>
          <div className="metric-badge">
            <div className="font-mono font-bold text-lg text-chart-3">
              {100 - splitRatio}%
            </div>
            <div className="text-[10px] text-muted-foreground">Testing</div>
            <div className="text-[10px] text-muted-foreground font-mono">
              {Math.round((dataset.length * (100 - splitRatio)) / 100)} rows
            </div>
          </div>
        </div>
      </div>

      {/* Train button */}
      <Button
        data-ocid="model.train.primary_button"
        onClick={trainModel}
        disabled={isTraining}
        className="w-full h-12 bg-primary text-primary-foreground hover:bg-primary/90 font-display font-semibold text-sm rounded-xl"
      >
        {isTraining ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Training {selectedModel}...
          </>
        ) : (
          <>
            <Cpu className="w-4 h-4 mr-2" />
            Train Model
          </>
        )}
      </Button>

      {/* Training progress */}
      {isTraining && (
        <div
          data-ocid="model.training.loading_state"
          className="glass-card rounded-2xl p-4 space-y-3"
        >
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">Training progress</span>
            <span className="font-mono text-primary font-semibold">
              {trainingProgress}%
            </span>
          </div>
          <Progress value={trainingProgress} className="h-2.5" />
          <div className="flex items-center gap-2">
            <Loader2 className="w-3 h-3 text-primary animate-spin" />
            <span className="text-xs text-muted-foreground animate-pulse">
              Fitting {selectedModel} on training data...
            </span>
          </div>
        </div>
      )}

      {/* Model metrics */}
      {modelMetrics && trainedModel && !isTraining && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          data-ocid="model.metrics.success_state"
          className="glass-card rounded-2xl p-4 space-y-3"
        >
          <div className="flex items-center justify-between">
            <div className="font-display font-semibold text-sm text-foreground">
              {trainedModel}
            </div>
            <Badge className="bg-green-500/10 text-green-400 border-green-500/20 text-xs">
              ✓ Trained
            </Badge>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[
              {
                label: "R² Score",
                value: modelMetrics.r2.toFixed(4),
                color: "text-primary",
              },
              {
                label: "Accuracy",
                value: `${(modelMetrics.accuracy * 100).toFixed(1)}%`,
                color: "text-chart-2",
              },
              {
                label: "MAE",
                value: `$${Math.round(modelMetrics.mae).toLocaleString()}`,
                color: "text-foreground",
              },
              {
                label: "MSE",
                value:
                  modelMetrics.mse.toFixed(0).length > 8
                    ? `${(modelMetrics.mse / 1e9).toFixed(2)}B`
                    : modelMetrics.mse.toFixed(0),
                color: "text-foreground",
              },
            ].map((m) => (
              <div key={m.label} className="metric-badge">
                <div className={`font-mono font-bold text-base ${m.color}`}>
                  {m.value}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {m.label}
                </div>
              </div>
            ))}
          </div>

          {/* Feature importance chart */}
          <div className="pt-1">
            <div className="text-xs text-muted-foreground mb-2">
              Feature Importance
            </div>
            {importance.map((fi) => (
              <div key={fi.feature} className="flex items-center gap-2 mb-1.5">
                <span className="font-mono text-[11px] text-muted-foreground w-16">
                  {fi.feature}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-muted/30 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${fi.importance * 100}%` }}
                    transition={{ delay: 0.2, duration: 0.6, ease: "easeOut" }}
                    className="h-full rounded-full bg-primary"
                  />
                </div>
                <span className="font-mono text-[11px] text-primary w-10 text-right">
                  {(fi.importance * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>

          {/* Dataset info */}
          <div className="text-xs text-muted-foreground pt-1 border-t border-border/30">
            Trained on {Math.round((dataset.length * splitRatio) / 100)} rows ·
            Avg market price ${Math.round(avgPrice).toLocaleString()}
          </div>
        </motion.div>
      )}

      {!trainedModel && !isTraining && (
        <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
          <Cpu className="w-12 h-12 mb-3 opacity-20" />
          <p className="text-sm">No model trained yet</p>
          <p className="text-xs mt-1">
            Select an algorithm and tap Train Model
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Predict Tab ──────────────────────────────────────────────────────────────
type PredictSubTab = "predict" | "history";

function PredictTab({
  dataset,
  trainedModel,
  onNavigateModel,
}: {
  dataset: DataRow[];
  trainedModel: ModelType | null;
  onNavigateModel: () => void;
}) {
  const [subTab, setSubTab] = useState<PredictSubTab>("predict");

  return (
    <div className="flex flex-col min-h-full">
      <div className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm px-4 pt-4 pb-3 border-b border-border/30">
        <div className="flex gap-2">
          {(["predict", "history"] as PredictSubTab[]).map((t) => (
            <button
              key={t}
              type="button"
              data-ocid={`predict.${t}.tab`}
              onClick={() => setSubTab(t)}
              className={`sub-tab capitalize flex-1 ${subTab === t ? "active" : ""}`}
            >
              {t === "predict" ? "Predict" : "History"}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 px-4 py-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={subTab}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.15 }}
          >
            {subTab === "predict" && (
              <PredictView
                dataset={dataset}
                trainedModel={trainedModel}
                onNavigateModel={onNavigateModel}
              />
            )}
            {subTab === "history" && <HistoryView />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function PredictView({
  dataset,
  trainedModel,
  onNavigateModel,
}: {
  dataset: DataRow[];
  trainedModel: ModelType | null;
  onNavigateModel: () => void;
}) {
  const [sqft, setSqft] = useState("1800");
  const [bedrooms, setBedrooms] = useState("3");
  const [bathrooms, setBathrooms] = useState("2");
  const [age, setAge] = useState("10");
  const [result, setResult] = useState<{
    price: number;
    confidence: number;
  } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const saveMutation = useSavePrediction();

  const handlePredict = async () => {
    if (!trainedModel) {
      toast.error("Train a model first!");
      return;
    }
    setIsPredicting(true);
    await new Promise((r) => setTimeout(r, 400));
    const res = predictPrice(
      dataset,
      { sqft: +sqft, bedrooms: +bedrooms, bathrooms: +bathrooms, age: +age },
      trainedModel,
    );
    setResult(res);
    setIsPredicting(false);
    const features = JSON.stringify({
      sqft: +sqft,
      bedrooms: +bedrooms,
      bathrooms: +bathrooms,
      age: +age,
    });
    saveMutation.mutate(
      {
        modelName: trainedModel,
        inputFeatures: features,
        predictedValue: res.price,
        confidence: res.confidence,
      },
      {
        onSuccess: () => toast.success("Saved to history"),
        onError: () => toast.error("Failed to save"),
      },
    );
  };

  return (
    <div className="space-y-4">
      {!trainedModel && (
        <div
          data-ocid="predictions.model.error_state"
          className="rounded-2xl border border-destructive/30 bg-destructive/8 p-4 flex items-center gap-3"
        >
          <div className="flex-1 text-sm text-destructive/90">
            No model trained yet
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={onNavigateModel}
            className="border-border h-9 text-xs"
            data-ocid="predictions.train_now.button"
          >
            Train Now
          </Button>
        </div>
      )}

      {/* Input form */}
      <div className="glass-card rounded-2xl p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="font-display font-semibold text-sm text-foreground">
            Input Features
          </div>
          {trainedModel && (
            <Badge className="bg-primary/10 text-primary border-primary/20 text-xs">
              {trainedModel.split(" ")[0]}
            </Badge>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Square Footage</div>
            <Input
              id="pred-sqft"
              data-ocid="predictions.sqft.input"
              type="number"
              value={sqft}
              onChange={(e) => setSqft(e.target.value)}
              className="h-11 bg-background/60 border-border/60 font-mono"
              placeholder="1800"
            />
          </div>
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Bedrooms</div>
            <Input
              id="pred-bedrooms"
              data-ocid="predictions.bedrooms.input"
              type="number"
              value={bedrooms}
              onChange={(e) => setBedrooms(e.target.value)}
              className="h-11 bg-background/60 border-border/60 font-mono"
              placeholder="3"
            />
          </div>
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Bathrooms</div>
            <Input
              id="pred-bathrooms"
              data-ocid="predictions.bathrooms.input"
              type="number"
              value={bathrooms}
              onChange={(e) => setBathrooms(e.target.value)}
              className="h-11 bg-background/60 border-border/60 font-mono"
              placeholder="2"
            />
          </div>
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">House Age (yrs)</div>
            <Input
              id="pred-age"
              data-ocid="predictions.age.input"
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="h-11 bg-background/60 border-border/60 font-mono"
              placeholder="10"
            />
          </div>
        </div>

        <Button
          data-ocid="predictions.predict.primary_button"
          onClick={handlePredict}
          disabled={isPredicting || !trainedModel}
          className="w-full h-12 bg-primary text-primary-foreground hover:bg-primary/90 font-display font-semibold rounded-xl"
        >
          {isPredicting ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Predicting...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" /> Predict Price
            </>
          )}
        </Button>
      </div>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            data-ocid="predictions.result.success_state"
            className="relative rounded-2xl overflow-hidden p-5 text-center glow-border bg-card"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/3 pointer-events-none" />
            <div className="relative space-y-3">
              <div className="text-xs text-muted-foreground uppercase tracking-widest">
                Predicted Price
              </div>
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200 }}
                className="font-display text-4xl font-bold text-primary"
              >
                ${result.price.toLocaleString()}
              </motion.div>
              <div className="flex items-center justify-center gap-3">
                <span className="text-xs text-muted-foreground">
                  Confidence
                </span>
                <Progress
                  value={result.confidence * 100}
                  className="h-2 w-24"
                />
                <span className="font-mono text-sm text-chart-2 font-semibold">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function HistoryView() {
  const { data: predictions, isLoading } = useGetPredictions();
  const clearMutation = useClearPredictions();

  const downloadCSV = () => {
    if (!predictions?.length) return;
    const header = "Timestamp,Model,Features,Predicted Price,Confidence";
    const rows = predictions.map(
      (p) =>
        `${new Date(Number(p.timestamp)).toISOString()},"${p.modelName}","${p.inputFeatures}",${p.predictedValue},${(p.confidence * 100).toFixed(1)}%`,
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ml_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("CSV downloaded");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="font-mono text-xs">
          {predictions?.length ?? 0} records
        </Badge>
        <div className="ml-auto flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            data-ocid="history.download.button"
            onClick={downloadCSV}
            disabled={!predictions?.length}
            className="h-9 text-xs border border-border/50 text-muted-foreground hover:text-foreground"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" /> CSV
          </Button>
          <Button
            variant="ghost"
            size="sm"
            data-ocid="history.clear.delete_button"
            onClick={() =>
              clearMutation.mutate(undefined, {
                onSuccess: () => toast.success("History cleared"),
              })
            }
            disabled={clearMutation.isPending || !predictions?.length}
            className="h-9 text-xs border border-destructive/30 text-destructive hover:bg-destructive/10"
          >
            {clearMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Trash2 className="w-3.5 h-3.5 mr-1.5" />
            )}
            Clear
          </Button>
        </div>
      </div>

      {isLoading && (
        <div
          data-ocid="history.table.loading_state"
          className="flex items-center justify-center py-16 gap-2 text-muted-foreground"
        >
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading history...</span>
        </div>
      )}

      {!isLoading && (!predictions || predictions.length === 0) && (
        <div
          data-ocid="history.table.empty_state"
          className="flex flex-col items-center justify-center py-16 text-muted-foreground"
        >
          <History className="w-12 h-12 mb-3 opacity-20" />
          <p className="text-sm">No predictions yet</p>
          <p className="text-xs mt-1">Run a prediction to see history</p>
        </div>
      )}

      {predictions && predictions.length > 0 && (
        <div className="space-y-2">
          {predictions.map((p, i) => {
            let features: Record<string, number> = {};
            try {
              features = JSON.parse(p.inputFeatures);
            } catch {
              /* ignore */
            }
            return (
              <motion.div
                key={`ph-${i}-${String(p.timestamp)}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                data-ocid={`history.row.item.${i + 1}`}
                className="glass-card rounded-xl p-3 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <Badge className="bg-primary/10 text-primary border-primary/20 text-xs">
                    {p.modelName.split(" ")[0]}
                  </Badge>
                  <span className="font-display font-bold text-base text-chart-2">
                    ${p.predictedValue.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Progress
                    value={p.confidence * 100}
                    className="h-1.5 flex-1"
                  />
                  <span className="font-mono text-xs text-muted-foreground">
                    {(p.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {Object.entries(features)
                      .map(([k, v]) => `${k}:${v}`)
                      .join(" · ")}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {new Date(Number(p.timestamp)).toLocaleDateString()}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── More Tab ─────────────────────────────────────────────────────────────────
type MoreSubTab = "download" | "guide" | "help" | "about";

function MoreTab() {
  const [subTab, setSubTab] = useState<MoreSubTab>("guide");

  return (
    <div className="flex flex-col min-h-full">
      <div className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm px-4 pt-4 pb-3 border-b border-border/30">
        <div className="flex gap-2 overflow-x-auto custom-scroll pb-0.5">
          {(["guide", "help", "about", "download"] as MoreSubTab[]).map((t) => (
            <button
              key={t}
              type="button"
              data-ocid={`more.${t}.tab`}
              onClick={() => setSubTab(t)}
              className={`sub-tab capitalize flex-shrink-0 ${subTab === t ? "active" : ""}`}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 px-4 py-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={subTab}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.15 }}
          >
            {subTab === "guide" && <WorkflowView />}
            {subTab === "help" && <HelpView />}
            {subTab === "about" && <AboutView />}
            {subTab === "download" && <DownloadView />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function DownloadView() {
  const { data: predictions } = useGetPredictions();

  const downloadCSV = () => {
    if (!predictions?.length) return;
    const header = "Timestamp,Model,Features,Predicted Price,Confidence";
    const rows = predictions.map(
      (p) =>
        `${new Date(Number(p.timestamp)).toISOString()},"${p.modelName}","${p.inputFeatures}",${p.predictedValue},${(p.confidence * 100).toFixed(1)}%`,
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ml_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Downloaded predictions CSV");
  };

  return (
    <div className="space-y-4">
      <div className="glass-card rounded-2xl p-4 space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-chart-2/15 border border-chart-2/25 flex items-center justify-center">
            <Download className="w-5 h-5 text-chart-2" />
          </div>
          <div>
            <div className="font-display font-semibold text-sm text-foreground">
              Export Predictions
            </div>
            <div className="text-xs text-muted-foreground">
              {predictions?.length ?? 0} records available
            </div>
          </div>
        </div>
        <Button
          data-ocid="download.csv.primary_button"
          onClick={downloadCSV}
          disabled={!predictions?.length}
          className="w-full h-11 bg-chart-2/10 hover:bg-chart-2/20 text-chart-2 border border-chart-2/25 hover:border-chart-2/40"
          variant="ghost"
        >
          <Download className="w-4 h-4 mr-2" />
          Download CSV
        </Button>
      </div>

      <div className="glass-card rounded-2xl p-4 space-y-2">
        <div className="font-display font-semibold text-sm text-foreground">
          CSV Format
        </div>
        <div className="font-mono text-xs text-muted-foreground bg-muted/20 rounded-lg p-3 space-y-1">
          <div className="text-primary">
            Timestamp, Model, Features, Price, Confidence
          </div>
          <div>2024-01-15T10:30:00Z, "Random Forest", ...</div>
          <div>...</div>
        </div>
        <p className="text-xs text-muted-foreground">
          Compatible with Excel, Python (pandas), R, and any CSV viewer.
        </p>
      </div>
    </div>
  );
}

function WorkflowView() {
  const steps = [
    {
      n: 1,
      title: "Explore Dataset",
      desc: "Review the 20-row housing dataset or upload your own CSV with columns: sqft, bedrooms, bathrooms, age, price.",
      icon: Database,
    },
    {
      n: 2,
      title: "Check Summary",
      desc: "View statistical summaries — mean, median, std dev — to understand data distribution and quality.",
      icon: BarChart2,
    },
    {
      n: 3,
      title: "Visualize Patterns",
      desc: "Use bar, line, and scatter charts to explore correlations between features.",
      icon: TrendingUp,
    },
    {
      n: 4,
      title: "Train a Model",
      desc: "Select an algorithm (Linear Regression, Decision Tree, Random Forest), set the split, and train.",
      icon: Brain,
    },
    {
      n: 5,
      title: "Make Predictions",
      desc: "Input house features to get an instant price prediction with a confidence score.",
      icon: Zap,
    },
    {
      n: 6,
      title: "Export Results",
      desc: "Download prediction history as CSV for further analysis.",
      icon: Download,
    },
  ];

  return (
    <div className="space-y-3">
      {steps.map((step, i) => (
        <motion.div
          key={step.n}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.06 }}
          data-ocid={`guide.step.item.${step.n}`}
          className="flex gap-3 glass-card rounded-xl p-3.5"
        >
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/15 border border-primary/25 flex items-center justify-center">
            <span className="font-display font-bold text-primary text-xs">
              {step.n}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <step.icon className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
              <span className="font-display font-semibold text-sm text-foreground">
                {step.title}
              </span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {step.desc}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function HelpView() {
  const faqs = [
    {
      q: "What CSV format is required?",
      a: "Your CSV must have lowercase headers: sqft, bedrooms, bathrooms, age, price. Numeric values only, no currency symbols.",
    },
    {
      q: "How is prediction calculated?",
      a: "Client-side multi-feature regression simulation using feature correlations from your dataset. No server needed.",
    },
    {
      q: "What do model metrics mean?",
      a: "R² (closer to 1 = better fit), MSE (mean squared error — lower is better), MAE (mean absolute error in $), Accuracy (% within margin).",
    },
    {
      q: "Where is history stored?",
      a: "On-chain in an Internet Computer canister — tamper-proof and persistent across page refreshes.",
    },
    {
      q: "Why do metrics change on retrain?",
      a: "A small random variance is applied to simulate real-world training variance. Differences are minor and realistic.",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-primary/20 bg-primary/5 p-4">
        <div className="font-display font-semibold text-sm text-primary mb-2">
          Quick Start
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Go to <strong className="text-foreground">Train</strong> tab → select
          algorithm → tap{" "}
          <strong className="text-foreground">Train Model</strong>. Then go to{" "}
          <strong className="text-foreground">Predict</strong> tab to get price
          estimates.
        </p>
      </div>

      <Accordion type="single" collapsible className="space-y-2">
        {faqs.map((faq, i) => (
          <AccordionItem
            key={faq.q.slice(0, 20)}
            value={`faq-${i}`}
            data-ocid={`help.faq.item.${i + 1}`}
            className="glass-card rounded-xl overflow-hidden px-4 border-none"
          >
            <AccordionTrigger className="text-sm font-medium text-foreground hover:no-underline py-3.5">
              {faq.q}
            </AccordionTrigger>
            <AccordionContent className="text-xs text-muted-foreground pb-3.5 leading-relaxed">
              {faq.a}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}

function AboutView() {
  const stack = [
    { name: "React 19", desc: "UI" },
    { name: "TypeScript", desc: "Types" },
    { name: "Recharts", desc: "Charts" },
    { name: "ICP", desc: "Storage" },
    { name: "TanStack", desc: "Queries" },
    { name: "shadcn/ui", desc: "Components" },
    { name: "Tailwind", desc: "Styles" },
    { name: "Motion", desc: "Animations" },
  ];

  return (
    <div className="space-y-4">
      <div className="glass-card rounded-2xl p-5 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-chart-2/5 pointer-events-none" />
        <div className="relative flex gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary/25 to-chart-2/25 border border-primary/20 flex items-center justify-center flex-shrink-0">
            <Brain className="w-7 h-7 text-primary" />
          </div>
          <div className="flex-1">
            <div className="font-display font-bold text-base text-foreground mb-1">
              ML Studio
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Educational ML tools that run entirely in the browser — no Python
              environment required. Democratizing machine learning through
              visual, interactive, blockchain-native experiences.
            </p>
            <div className="flex flex-wrap gap-1.5 mt-3">
              {["Machine Learning", "Data Science", "Web3", "Education"].map(
                (t) => (
                  <Badge
                    key={t}
                    variant="outline"
                    className="border-border/50 text-xs"
                  >
                    {t}
                  </Badge>
                ),
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="font-display font-semibold text-sm text-foreground">
          Tech Stack
        </div>
        <div className="grid grid-cols-4 gap-2">
          {stack.map((t) => (
            <div
              key={t.name}
              className="metric-badge hover:border-primary/30 transition-colors cursor-default"
            >
              <div className="font-display font-semibold text-xs text-foreground">
                {t.name}
              </div>
              <div className="text-[10px] text-muted-foreground">{t.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card rounded-2xl p-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          ML Studio is open-source educational software. ML simulations are for
          demonstration purposes and should not be used for financial decisions.
          All predictions are generated client-side.
        </p>
      </div>

      <div className="text-center text-[11px] text-muted-foreground pt-2">
        © {new Date().getFullYear()}. Built with ❤️ using{" "}
        <a
          href={`https://caffeine.ai?utm_source=caffeine-footer&utm_medium=referral&utm_content=${encodeURIComponent(window.location.hostname)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary"
        >
          caffeine.ai
        </a>
      </div>
    </div>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MLStudioApp />
      <Toaster richColors position="bottom-center" />
    </QueryClientProvider>
  );
}
