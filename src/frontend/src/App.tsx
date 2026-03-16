import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Toaster } from "@/components/ui/sonner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Activity,
  AlertCircle,
  BarChart2,
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  FileText,
  FlaskConical,
  History,
  Info,
  Layers,
  LayoutDashboard,
  Loader2,
  RefreshCw,
  Search,
  Target,
  Trash2,
  TrendingUp,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
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
type Section =
  | "dashboard"
  | "upload"
  | "explore"
  | "visualize"
  | "train"
  | "predict"
  | "history"
  | "guide";

interface DataRow {
  [key: string]: string | number;
}

interface ModelMetrics {
  r2: number;
  mae: number;
  rmse: number;
  trainSamples: number;
  coefficients: Record<string, number>;
  intercept: number;
  featureMeans: Record<string, number>;
  featureStds: Record<string, number>;
  targetMean: number;
  targetStd: number;
}

interface ComparisonModel {
  name: string;
  r2: number;
  mae: number;
  rmse: number;
  alpha?: number;
}

// ─── Sample Housing Dataset ───────────────────────────────────────────────────
const SAMPLE_DATA: DataRow[] = [
  { sqft: 1200, bedrooms: 2, bathrooms: 1, age: 15, garage: 0, price: 185000 },
  { sqft: 1800, bedrooms: 3, bathrooms: 2, age: 8, garage: 1, price: 275000 },
  { sqft: 2400, bedrooms: 4, bathrooms: 2, age: 5, garage: 2, price: 365000 },
  { sqft: 950, bedrooms: 1, bathrooms: 1, age: 22, garage: 0, price: 145000 },
  { sqft: 3200, bedrooms: 5, bathrooms: 3, age: 3, garage: 2, price: 490000 },
  { sqft: 1600, bedrooms: 3, bathrooms: 2, age: 12, garage: 1, price: 245000 },
  { sqft: 2100, bedrooms: 4, bathrooms: 2, age: 7, garage: 1, price: 320000 },
  { sqft: 1400, bedrooms: 2, bathrooms: 1, age: 18, garage: 0, price: 210000 },
  { sqft: 2800, bedrooms: 4, bathrooms: 3, age: 2, garage: 2, price: 425000 },
  { sqft: 1100, bedrooms: 2, bathrooms: 1, age: 30, garage: 0, price: 160000 },
  { sqft: 3600, bedrooms: 5, bathrooms: 4, age: 1, garage: 3, price: 560000 },
  { sqft: 1700, bedrooms: 3, bathrooms: 2, age: 10, garage: 1, price: 258000 },
  { sqft: 2250, bedrooms: 4, bathrooms: 3, age: 6, garage: 2, price: 345000 },
  { sqft: 1350, bedrooms: 2, bathrooms: 2, age: 14, garage: 1, price: 220000 },
  { sqft: 4000, bedrooms: 6, bathrooms: 4, age: 4, garage: 3, price: 620000 },
  { sqft: 1050, bedrooms: 2, bathrooms: 1, age: 25, garage: 0, price: 155000 },
  { sqft: 2600, bedrooms: 4, bathrooms: 3, age: 9, garage: 2, price: 398000 },
  { sqft: 1900, bedrooms: 3, bathrooms: 2, age: 11, garage: 1, price: 292000 },
  { sqft: 3000, bedrooms: 5, bathrooms: 3, age: 6, garage: 2, price: 455000 },
  { sqft: 1500, bedrooms: 3, bathrooms: 1, age: 20, garage: 0, price: 228000 },
];

// ─── Linear Algebra Helpers ───────────────────────────────────────────────────
function transpose(matrix: number[][]): number[][] {
  if (!matrix.length) return [];
  return matrix[0].map((_, i) => matrix.map((row) => row[i]));
}

function matMul(A: number[][], B: number[][]): number[][] {
  const result: number[][] = Array.from({ length: A.length }, () =>
    new Array(B[0].length).fill(0),
  );
  for (let i = 0; i < A.length; i++)
    for (let k = 0; k < B.length; k++)
      for (let j = 0; j < B[0].length; j++) result[i][j] += A[i][k] * B[k][j];
  return result;
}

function invert2D(matrix: number[][]): number[][] | null {
  const n = matrix.length;
  const aug = matrix.map((row, i) => [
    ...row,
    ...new Array(n).fill(0).map((_, j) => (i === j ? 1 : 0)),
  ]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-12) return null;
    const div = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= div;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

function trainLinearRegression(
  X: number[][],
  y: number[],
  alpha = 0,
): number[] | null {
  // OLS with optional L2 ridge: (X'X + alpha*I)^-1 X'y
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  if (alpha > 0) for (let i = 0; i < XtX.length; i++) XtX[i][i] += alpha;
  const inv = invert2D(XtX);
  if (!inv) return null;
  const XtY = Xt.map((row) => row.reduce((sum, val, i) => sum + val * y[i], 0));
  return inv.map((row) => row.reduce((sum, val, i) => sum + val * XtY[i], 0));
}

function computeMetrics(
  predictions: number[],
  actuals: number[],
): { r2: number; mae: number; rmse: number } {
  const mean = actuals.reduce((a, b) => a + b, 0) / actuals.length;
  const ssTot = actuals.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ssRes = actuals.reduce((s, v, i) => s + (v - predictions[i]) ** 2, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const mae =
    predictions.reduce((s, v, i) => s + Math.abs(v - actuals[i]), 0) /
    actuals.length;
  const rmse = Math.sqrt(
    predictions.reduce((s, v, i) => s + (v - actuals[i]) ** 2, 0) /
      actuals.length,
  );
  return { r2, mae, rmse };
}

// ─── CSV Parser ───────────────────────────────────────────────────────────────
function parseCSV(text: string): DataRow[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0]
    .split(",")
    .map((h) => h.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
    const row: DataRow = {};
    for (let i = 0; i < headers.length; i++) {
      const h = headers[i];
      const num = Number.parseFloat(vals[i]);
      row[h] = Number.isNaN(num) ? vals[i] : num;
    }
    return row;
  });
}

function getNumericColumns(data: DataRow[]): string[] {
  if (!data.length) return [];
  return Object.keys(data[0]).filter((col) =>
    data.every(
      (row) =>
        row[col] !== undefined &&
        row[col] !== "" &&
        !Number.isNaN(Number(row[col])),
    ),
  );
}

function getMissingCount(data: DataRow[]): number {
  let count = 0;
  for (const row of data) {
    for (const v of Object.values(row)) {
      if (v === "" || v === null || v === undefined) count++;
    }
  }
  return count;
}

// ─── Sidebar Nav Items ────────────────────────────────────────────────────────
const NAV_ITEMS: {
  id: Section;
  label: string;
  icon: React.ReactNode;
  desc: string;
}[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: <LayoutDashboard size={18} />,
    desc: "Overview",
  },
  {
    id: "upload",
    label: "Upload Data",
    icon: <Upload size={18} />,
    desc: "CSV import",
  },
  {
    id: "explore",
    label: "Explore",
    icon: <Search size={18} />,
    desc: "Data preview",
  },
  {
    id: "visualize",
    label: "Visualize",
    icon: <BarChart2 size={18} />,
    desc: "Charts",
  },
  {
    id: "train",
    label: "Train Model",
    icon: <Brain size={18} />,
    desc: "Fit & evaluate",
  },
  {
    id: "predict",
    label: "Predict",
    icon: <Zap size={18} />,
    desc: "Run inference",
  },
  {
    id: "history",
    label: "History",
    icon: <History size={18} />,
    desc: "Past predictions",
  },
  { id: "guide", label: "Guide", icon: <BookOpen size={18} />, desc: "How-to" },
];

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({
  label,
  value,
  icon,
  color = "primary",
  sub,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color?: "primary" | "accent" | "green" | "purple";
  sub?: string;
}) {
  const colors = {
    primary: "text-primary",
    accent: "text-accent",
    green: "text-emerald-400",
    purple: "text-violet-400",
  };
  return (
    <div className="glass-card rounded-xl p-4 flex items-start gap-3">
      <div className={`mt-0.5 p-2 rounded-lg bg-secondary ${colors[color]}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground mb-0.5">{label}</p>
        <p className="text-xl font-bold font-mono text-foreground">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeSection, setActiveSection] = useState<Section>("dashboard");
  const [data, setData] = useState<DataRow[]>([]);
  const [datasetName, setDatasetName] = useState("");
  const [numericCols, setNumericCols] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState("");
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [model, setModel] = useState<ModelMetrics | null>(null);
  const [training, setTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [predInputs, setPredInputs] = useState<Record<string, string>>({});
  const [predResult, setPredResult] = useState<number | null>(null);
  const [scatterX, setScatterX] = useState("");
  const [scatterY, setScatterY] = useState("");
  const [histCol, setHistCol] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // ICP hooks
  const { data: predictions = [], isLoading: histLoading } =
    useGetPredictions();
  const savePrediction = useSavePrediction();
  const clearPredictions = useClearPredictions();

  // Detect mobile
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handler);
    // Auto-close sidebar on mobile
    if (window.innerWidth < 768) setSidebarOpen(false);
    return () => window.removeEventListener("resize", handler);
  }, []);

  // ─── Data loading ──────────────────────────────────────────────────────────
  const loadData = useCallback((rows: DataRow[], name: string) => {
    setData(rows);
    setDatasetName(name);
    const nums = getNumericColumns(rows);
    setNumericCols(nums);
    const defaultTarget = nums[nums.length - 1] ?? "";
    setTargetCol(defaultTarget);
    setFeatureCols(nums.filter((c) => c !== defaultTarget));
    setModel(null);
    setPredResult(null);
    setPredInputs({});
    if (nums.length >= 2) {
      setScatterX(nums[0]);
      setScatterY(defaultTarget);
      setHistCol(defaultTarget);
    }
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith(".csv")) {
        toast.error("Please upload a CSV file (.csv)");
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const rows = parseCSV(text);
        if (rows.length < 2) {
          toast.error("CSV must have at least 2 rows of data");
          return;
        }
        loadData(rows, file.name);
        toast.success(`Loaded ${rows.length} rows from ${file.name}`);
        setActiveSection("explore");
      };
      reader.readAsText(file);
    },
    [loadData],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  // ─── Training ─────────────────────────────────────────────────────────────
  const trainModel = useCallback(async () => {
    if (!data.length || !targetCol || !featureCols.length) return;
    setTraining(true);
    setTrainProgress(0);
    setPredResult(null);

    // Animate progress
    const steps = 20;
    for (let i = 1; i <= steps; i++) {
      await new Promise((r) => setTimeout(r, 75));
      setTrainProgress(Math.round((i / steps) * 100));
    }

    // Normalize features
    const means: Record<string, number> = {};
    const stds: Record<string, number> = {};
    for (const col of featureCols) {
      const vals = data.map((r) => Number(r[col]));
      means[col] = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance =
        vals.reduce((s, v) => s + (v - means[col]) ** 2, 0) / vals.length;
      stds[col] = Math.sqrt(variance) || 1;
    }

    const yVals = data.map((r) => Number(r[targetCol]));
    const yMean = yVals.reduce((a, b) => a + b, 0) / yVals.length;
    const yVariance =
      yVals.reduce((s, v) => s + (v - yMean) ** 2, 0) / yVals.length;
    const yStd = Math.sqrt(yVariance) || 1;

    // Build design matrix with intercept
    const X = data.map((row) => [
      1,
      ...featureCols.map((col) => (Number(row[col]) - means[col]) / stds[col]),
    ]);
    const y = yVals.map((v) => (v - yMean) / yStd);

    const beta = trainLinearRegression(X, y, 0);
    if (!beta) {
      toast.error(
        "Training failed — matrix is singular. Try removing correlated columns.",
      );
      setTraining(false);
      return;
    }

    // Denormalize coefficients
    const intercept =
      beta[0] * yStd +
      yMean -
      featureCols.reduce(
        (s, col, i) => s + (beta[i + 1] * yStd * means[col]) / stds[col],
        0,
      );
    const coefficients: Record<string, number> = {};
    featureCols.forEach((col, i) => {
      coefficients[col] = (beta[i + 1] * yStd) / stds[col];
    });

    // Compute metrics
    const preds = data.map((row) => {
      return (
        intercept +
        featureCols.reduce(
          (s, col) => s + coefficients[col] * Number(row[col]),
          0,
        )
      );
    });
    const { r2, mae, rmse } = computeMetrics(preds, yVals);

    setModel({
      r2,
      mae,
      rmse,
      trainSamples: data.length,
      coefficients,
      intercept,
      featureMeans: means,
      featureStds: stds,
      targetMean: yMean,
      targetStd: yStd,
    });

    // Init prediction inputs
    const initInputs: Record<string, string> = {};
    for (const col of featureCols) {
      initInputs[col] = String(means[col].toFixed(2));
    }
    setPredInputs(initInputs);

    setTraining(false);
    toast.success("Model trained successfully!");
    setActiveSection("predict");
  }, [data, targetCol, featureCols]);

  // ─── Prediction ───────────────────────────────────────────────────────────
  const runPrediction = useCallback(() => {
    if (!model) return;
    const pred =
      model.intercept +
      featureCols.reduce(
        (s, col) => s + model.coefficients[col] * Number(predInputs[col] || 0),
        0,
      );
    setPredResult(pred);
  }, [model, featureCols, predInputs]);

  const savePred = useCallback(async () => {
    if (predResult === null) return;
    const inputStr = featureCols
      .map((c) => `${c}=${predInputs[c] ?? 0}`)
      .join(", ");
    const confidence = 0.9;
    await savePrediction.mutateAsync({
      modelName: `Linear Regression (${targetCol})`,
      inputFeatures: inputStr,
      predictedValue: predResult,
      confidence,
    });
    toast.success("Prediction saved to history!");
  }, [predResult, featureCols, predInputs, targetCol, savePrediction]);

  // ─── Model comparison ─────────────────────────────────────────────────────
  const modelComparison: ComparisonModel[] = (() => {
    if (!data.length || !targetCol || !featureCols.length) return [];
    const yVals = data.map((r) => Number(r[targetCol]));
    const means: Record<string, number> = {};
    const stds: Record<string, number> = {};
    for (const col of featureCols) {
      const vals = data.map((r) => Number(r[col]));
      means[col] = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance =
        vals.reduce((s, v) => s + (v - means[col]) ** 2, 0) / vals.length;
      stds[col] = Math.sqrt(variance) || 1;
    }
    const yMean = yVals.reduce((a, b) => a + b, 0) / yVals.length;
    const yStd =
      Math.sqrt(
        yVals.reduce((s, v) => s + (v - yMean) ** 2, 0) / yVals.length,
      ) || 1;
    const X = data.map((row) => [
      1,
      ...featureCols.map((col) => (Number(row[col]) - means[col]) / stds[col]),
    ]);
    const y = yVals.map((v) => (v - yMean) / yStd);
    const models: ComparisonModel[] = [];
    for (const [name, alpha] of [
      ["Linear Regression", 0],
      ["Ridge (α=1)", 1],
      ["Ridge (α=10)", 10],
    ] as [string, number][]) {
      const beta = trainLinearRegression(X, y, alpha);
      if (!beta) continue;
      const preds = X.map(
        (row) => row.reduce((s, v, i) => s + beta[i] * v, 0) * yStd + yMean,
      );
      const m = computeMetrics(preds, yVals);
      models.push({ name, r2: m.r2, mae: m.mae, rmse: m.rmse, alpha });
    }
    return models;
  })();

  // ─── Correlation ──────────────────────────────────────────────────────────
  const correlationData = (() => {
    if (!data.length || !targetCol || !featureCols.length) return [];
    const yVals = data.map((r) => Number(r[targetCol]));
    const yMean = yVals.reduce((a, b) => a + b, 0) / yVals.length;
    return featureCols.map((col) => {
      const xVals = data.map((r) => Number(r[col]));
      const xMean = xVals.reduce((a, b) => a + b, 0) / xVals.length;
      const cov = xVals.reduce(
        (s, v, i) => s + (v - xMean) * (yVals[i] - yMean),
        0,
      );
      const stdX = Math.sqrt(xVals.reduce((s, v) => s + (v - xMean) ** 2, 0));
      const stdY = Math.sqrt(yVals.reduce((s, v) => s + (v - yMean) ** 2, 0));
      const pearson = stdX * stdY === 0 ? 0 : cov / (stdX * stdY);
      return {
        feature: col,
        correlation: Number.parseFloat(pearson.toFixed(4)),
      };
    });
  })();

  // ─── Histogram data ───────────────────────────────────────────────────────
  const histogramData = (() => {
    if (!data.length || !histCol) return [];
    const vals = data
      .map((r) => Number(r[histCol]))
      .filter((v) => !Number.isNaN(v));
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const bins = Math.min(20, Math.ceil(Math.sqrt(vals.length)));
    const binSize = (max - min) / bins || 1;
    const counts: Record<number, number> = {};
    for (const v of vals) {
      const bin = Math.floor((v - min) / binSize);
      counts[bin] = (counts[bin] || 0) + 1;
    }
    return Array.from({ length: bins }, (_, i) => ({
      bin: (min + i * binSize).toFixed(1),
      count: counts[i] || 0,
    }));
  })();

  // ─── Scatter data ─────────────────────────────────────────────────────────
  const scatterData = (() => {
    if (!data.length || !scatterX || !scatterY) return [];
    return data.slice(0, 500).map((row) => ({
      x: Number(row[scatterX]),
      y: Number(row[scatterY]),
    }));
  })();

  // ─── Feature importance ───────────────────────────────────────────────────
  const featureImportance = model
    ? Object.entries(model.coefficients)
        .map(([feat, coef]) => ({ feature: feat, importance: Math.abs(coef) }))
        .sort((a, b) => b.importance - a.importance)
    : [];

  // ─── Status ───────────────────────────────────────────────────────────────
  const status = model ? "trained" : data.length ? "loaded" : "empty";
  const statusBadge = {
    empty: { label: "No Data", className: "bg-muted text-muted-foreground" },
    loaded: {
      label: "Data Loaded",
      className: "bg-amber-500/20 text-amber-300",
    },
    trained: {
      label: "Model Trained",
      className: "bg-emerald-500/20 text-emerald-300",
    },
  }[status];

  // ─── Nav handler ─────────────────────────────────────────────────────────
  const navigate = (section: Section) => {
    setActiveSection(section);
    if (isMobile) setSidebarOpen(false);
  };

  // ──────────────────────────────────────────────────────────────────────────
  // RENDER SECTIONS
  // ──────────────────────────────────────────────────────────────────────────

  const renderDashboard = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Welcome to ML Studio
        </h2>
        <p className="text-muted-foreground">
          Upload any CSV dataset to explore, visualize, train, and predict with
          machine learning.
        </p>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Dataset Rows"
          value={data.length || "—"}
          icon={<Database size={16} />}
          color="primary"
        />
        <StatCard
          label="Columns"
          value={data.length ? Object.keys(data[0]).length : "—"}
          icon={<Layers size={16} />}
          color="accent"
        />
        <StatCard
          label="Numeric Cols"
          value={numericCols.length || "—"}
          icon={<Activity size={16} />}
          color="green"
        />
        <StatCard
          label="Model R²"
          value={model ? model.r2.toFixed(4) : "—"}
          icon={<TrendingUp size={16} />}
          color="purple"
          sub={model ? "Goodness of fit" : "Train first"}
        />
      </div>

      {/* Quick workflow steps */}
      <div className="glass-card rounded-xl p-5">
        <h3 className="font-display font-semibold text-foreground mb-4">
          ML Workflow
        </h3>
        <div className="space-y-3">
          {[
            {
              step: "01",
              label: "Upload CSV",
              desc: "Any dataset with numeric columns",
              done: data.length > 0,
              target: "upload" as Section,
            },
            {
              step: "02",
              label: "Explore Data",
              desc: "Preview rows and statistics",
              done: data.length > 0,
              target: "explore" as Section,
            },
            {
              step: "03",
              label: "Visualize",
              desc: "Charts, distributions, correlations",
              done: data.length > 0,
              target: "visualize" as Section,
            },
            {
              step: "04",
              label: "Train Model",
              desc: "Select target and fit regression",
              done: !!model,
              target: "train" as Section,
            },
            {
              step: "05",
              label: "Predict",
              desc: "Enter features and get predictions",
              done: false,
              target: "predict" as Section,
            },
          ].map((item) => (
            <button
              type="button"
              key={item.step}
              onClick={() => navigate(item.target)}
              data-ocid={`dashboard.${item.target}.link`}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-secondary/60 transition-colors text-left"
            >
              <span
                className={`text-xs font-mono font-bold w-8 shrink-0 ${
                  item.done ? "text-emerald-400" : "text-muted-foreground"
                }`}
              >
                {item.done ? (
                  <CheckCircle2 size={16} className="text-emerald-400" />
                ) : (
                  item.step
                )}
              </span>
              <div>
                <p className="text-sm font-medium text-foreground">
                  {item.label}
                </p>
                <p className="text-xs text-muted-foreground">{item.desc}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Recent predictions summary */}
      {predictions.length > 0 && (
        <div className="glass-card rounded-xl p-5">
          <h3 className="font-display font-semibold text-foreground mb-3">
            Recent Predictions
          </h3>
          <div className="space-y-2">
            {predictions.slice(0, 4).map((p, i) => (
              <div
                key={`pred-${i}-${p.timestamp}`}
                className="flex items-center justify-between py-2 border-b border-border last:border-0"
              >
                <div className="min-w-0">
                  <p className="text-sm text-foreground font-medium truncate">
                    {p.inputFeatures}
                  </p>
                  <p className="text-xs text-muted-foreground">{p.modelName}</p>
                </div>
                <Badge className="ml-2 shrink-0 font-mono">
                  {p.predictedValue.toFixed(2)}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderUpload = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Upload Dataset
        </h2>
        <p className="text-muted-foreground">
          Upload any CSV file. The app auto-detects all numeric columns as
          features.
        </p>
      </div>

      {/* Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") fileRef.current?.click();
        }}
        data-ocid="upload.dropzone"
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all ${
          isDragging
            ? "border-primary bg-primary/5 glow-border"
            : "border-border hover:border-primary/50 hover:bg-secondary/30"
        }`}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
          data-ocid="upload.input"
        />
        <motion.div
          animate={{ scale: isDragging ? 1.05 : 1 }}
          className="flex flex-col items-center gap-3"
        >
          <div className="p-4 rounded-full bg-primary/10 text-primary">
            <Upload size={32} />
          </div>
          <div>
            <p className="text-lg font-semibold text-foreground">
              Drop your CSV file here
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse — supports any CSV with numeric columns
            </p>
          </div>
          <div className="flex gap-2 flex-wrap justify-center mt-2">
            {["Housing", "Sales", "Medical", "Finance", "Any numeric CSV"].map(
              (t) => (
                <Badge key={t} variant="secondary" className="text-xs">
                  {t}
                </Badge>
              ),
            )}
          </div>
        </motion.div>
      </div>

      {/* Demo data button */}
      <div className="text-center">
        <p className="text-sm text-muted-foreground mb-3">
          — or try with sample data —
        </p>
        <Button
          variant="outline"
          onClick={() => {
            loadData(SAMPLE_DATA, "demo_housing.csv");
            toast.success("Demo housing dataset loaded!");
            setActiveSection("explore");
          }}
          data-ocid="upload.primary_button"
          className="gap-2"
        >
          <FlaskConical size={16} />
          Use Demo Dataset (Housing Prices)
        </Button>
      </div>

      {/* Format guide */}
      <div className="glass-card rounded-xl p-5">
        <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
          <Info size={16} className="text-primary" />
          Expected CSV Format
        </h3>
        <div className="overflow-x-auto">
          <pre className="font-mono text-xs text-muted-foreground leading-relaxed">
            {`sqft,bedrooms,bathrooms,age,price
1200,2,1,15,185000
1800,3,2,8,275000
2400,4,2,5,365000
...`}
          </pre>
        </div>
        <ul className="mt-3 space-y-1">
          {[
            "First row must be column headers",
            "Numeric columns are auto-detected as features",
            "You choose which column is the target (to predict)",
            "Missing/non-numeric values in numeric columns will cause errors",
          ].map((item) => (
            <li
              key={item}
              className="flex items-start gap-2 text-xs text-muted-foreground"
            >
              <CheckCircle2
                size={12}
                className="text-emerald-400 mt-0.5 shrink-0"
              />
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  const renderExplore = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Explore Dataset
        </h2>
        <p className="text-muted-foreground">
          Preview data, review statistics, and configure your target column.
        </p>
      </div>

      {!data.length ? (
        <EmptyState
          icon={<Database size={40} />}
          title="No dataset loaded"
          action={{
            label: "Upload CSV",
            onClick: () => setActiveSection("upload"),
          }}
        />
      ) : (
        <>
          {/* Dataset stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Total Rows"
              value={data.length}
              icon={<Database size={16} />}
            />
            <StatCard
              label="Total Cols"
              value={Object.keys(data[0]).length}
              icon={<Layers size={16} />}
              color="accent"
            />
            <StatCard
              label="Numeric Cols"
              value={numericCols.length}
              icon={<Activity size={16} />}
              color="green"
            />
            <StatCard
              label="Missing Values"
              value={getMissingCount(data)}
              icon={<AlertCircle size={16} />}
              color="purple"
            />
          </div>

          {/* Target column selector */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
              <Target size={16} className="text-primary" />
              Select Target Column (to predict)
            </h3>
            <div className="flex flex-wrap items-center gap-3">
              <Select
                value={targetCol}
                onValueChange={(v) => {
                  setTargetCol(v);
                  setFeatureCols(numericCols.filter((c) => c !== v));
                  setModel(null);
                  setPredResult(null);
                }}
              >
                <SelectTrigger
                  className="w-48"
                  data-ocid="explore.target.select"
                >
                  <SelectValue placeholder="Select target" />
                </SelectTrigger>
                <SelectContent>
                  {numericCols.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex flex-wrap gap-1">
                {featureCols.map((col) => (
                  <Badge
                    key={col}
                    variant="secondary"
                    className="text-xs font-mono"
                  >
                    {col}
                  </Badge>
                ))}
              </div>
            </div>
            {featureCols.length > 0 && (
              <p className="text-xs text-muted-foreground mt-2">
                {featureCols.length} feature
                {featureCols.length !== 1 ? "s" : ""} → predicting{" "}
                <strong className="text-foreground">{targetCol}</strong>
              </p>
            )}
          </div>

          {/* Column stats */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-3">
              Column Statistics
            </h3>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column</TableHead>
                    <TableHead>Min</TableHead>
                    <TableHead>Max</TableHead>
                    <TableHead>Mean</TableHead>
                    <TableHead>Std Dev</TableHead>
                    <TableHead>Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.keys(data[0]).map((col, i) => {
                    const isNum = numericCols.includes(col);
                    const vals = isNum ? data.map((r) => Number(r[col])) : [];
                    const mn = isNum ? Math.min(...vals) : null;
                    const mx = isNum ? Math.max(...vals) : null;
                    const mean = isNum
                      ? vals.reduce((a, b) => a + b, 0) / vals.length
                      : null;
                    const std = isNum
                      ? Math.sqrt(
                          vals.reduce((s, v) => s + (v - (mean ?? 0)) ** 2, 0) /
                            vals.length,
                        )
                      : null;
                    return (
                      <TableRow
                        key={col}
                        data-ocid={`explore.table.row.${i + 1}`}
                      >
                        <TableCell className="font-mono font-medium text-primary">
                          {col}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {mn !== null ? mn.toFixed(2) : "—"}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {mx !== null ? mx.toFixed(2) : "—"}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {mean !== null ? mean.toFixed(2) : "—"}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {std !== null ? std.toFixed(2) : "—"}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={`text-xs ${
                              isNum
                                ? "text-emerald-400 border-emerald-500/30"
                                : "text-amber-400 border-amber-500/30"
                            }`}
                          >
                            {isNum ? "numeric" : "categorical"}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Data preview */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-3">
              Data Preview (first 10 rows)
            </h3>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {Object.keys(data[0]).map((col) => (
                      <TableHead key={col} className="font-mono text-xs">
                        {col}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.slice(0, 10).map((row, i) => (
                    <TableRow
                      key={Object.values(row).slice(0, 3).join("-")}
                      data-ocid={`explore.preview.row.${i + 1}`}
                    >
                      {Object.entries(row).map(([colKey, val]) => (
                        <TableCell key={colKey} className="font-mono text-xs">
                          {typeof val === "number"
                            ? val.toLocaleString()
                            : String(val)}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </>
      )}
    </div>
  );

  const renderVisualize = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Visualize Data
        </h2>
        <p className="text-muted-foreground">
          Explore relationships, distributions, and correlations in your
          dataset.
        </p>
      </div>

      {!data.length ? (
        <EmptyState
          icon={<BarChart2 size={40} />}
          title="No dataset loaded"
          action={{
            label: "Upload CSV",
            onClick: () => setActiveSection("upload"),
          }}
        />
      ) : (
        <>
          {/* Scatter plot */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-3">Scatter Plot</h3>
            <div className="flex flex-wrap gap-3 mb-4">
              <div className="flex items-center gap-2">
                <Label className="text-xs">X axis</Label>
                <Select value={scatterX} onValueChange={setScatterX}>
                  <SelectTrigger
                    className="w-36 h-8 text-xs"
                    data-ocid="viz.scatter.x.select"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {numericCols.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs">Y axis</Label>
                <Select value={scatterY} onValueChange={setScatterY}>
                  <SelectTrigger
                    className="w-36 h-8 text-xs"
                    data-ocid="viz.scatter.y.select"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {numericCols.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="oklch(0.22 0.025 258)"
                />
                <XAxis
                  dataKey="x"
                  name={scatterX}
                  tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }}
                  label={{
                    value: scatterX,
                    position: "insideBottom",
                    offset: -5,
                    fill: "oklch(0.55 0.02 255)",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  dataKey="y"
                  name={scatterY}
                  tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }}
                  width={60}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    background: "oklch(0.13 0.025 258)",
                    border: "1px solid oklch(0.22 0.025 258)",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(val: number, name: string) => [
                    val.toFixed(2),
                    name,
                  ]}
                />
                <Scatter data={scatterData} data-ocid="viz.chart_point">
                  {scatterData.map((pt) => (
                    <Cell
                      key={`${pt.x}-${pt.y}`}
                      fill="oklch(0.72 0.19 197 / 0.7)"
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Histogram */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-3">Distribution</h3>
            <div className="flex items-center gap-2 mb-4">
              <Label className="text-xs">Column</Label>
              <Select value={histCol} onValueChange={setHistCol}>
                <SelectTrigger
                  className="w-40 h-8 text-xs"
                  data-ocid="viz.histogram.select"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {numericCols.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={histogramData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="oklch(0.22 0.025 258)"
                />
                <XAxis
                  dataKey="bin"
                  tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 10 }}
                />
                <YAxis tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    background: "oklch(0.13 0.025 258)",
                    border: "1px solid oklch(0.22 0.025 258)",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="oklch(0.78 0.19 60)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Correlation table */}
          {targetCol && (
            <div className="glass-card rounded-xl p-5">
              <h3 className="font-semibold text-foreground mb-3">
                Pearson Correlation with{" "}
                <span className="text-primary">{targetCol}</span>
              </h3>
              <div className="space-y-3">
                {correlationData
                  .sort(
                    (a, b) => Math.abs(b.correlation) - Math.abs(a.correlation),
                  )
                  .map((item) => (
                    <div key={item.feature} className="flex items-center gap-3">
                      <span className="font-mono text-xs w-28 shrink-0 text-muted-foreground">
                        {item.feature}
                      </span>
                      <div className="flex-1 bg-secondary rounded-full h-2 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{
                            width: `${Math.abs(item.correlation) * 100}%`,
                          }}
                          transition={{ duration: 0.6 }}
                          className={`h-full rounded-full ${
                            item.correlation >= 0
                              ? "bg-primary"
                              : "bg-destructive"
                          }`}
                        />
                      </div>
                      <span
                        className={`font-mono text-xs w-14 text-right ${
                          Math.abs(item.correlation) > 0.7
                            ? "text-emerald-400"
                            : Math.abs(item.correlation) > 0.4
                              ? "text-amber-400"
                              : "text-muted-foreground"
                        }`}
                      >
                        {item.correlation.toFixed(3)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );

  const renderTrain = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Train Model
        </h2>
        <p className="text-muted-foreground">
          Fit a linear regression model using all numeric features.
        </p>
      </div>

      {!data.length ? (
        <EmptyState
          icon={<Brain size={40} />}
          title="No dataset loaded"
          action={{
            label: "Upload CSV",
            onClick: () => setActiveSection("upload"),
          }}
        />
      ) : (
        <>
          {/* Config card */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-4">
              Training Configuration
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <Label className="text-xs mb-1.5 block">Target Column</Label>
                <Select
                  value={targetCol}
                  onValueChange={(v) => {
                    setTargetCol(v);
                    setFeatureCols(numericCols.filter((c) => c !== v));
                    setModel(null);
                    setPredResult(null);
                  }}
                >
                  <SelectTrigger data-ocid="train.target.select">
                    <SelectValue placeholder="Select target" />
                  </SelectTrigger>
                  <SelectContent>
                    {numericCols.map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs mb-1.5 block">Feature Columns</Label>
                <div className="p-2.5 rounded-lg bg-secondary/50 border border-border min-h-[40px] flex flex-wrap gap-1">
                  {featureCols.map((col) => (
                    <Badge
                      key={col}
                      variant="secondary"
                      className="text-xs font-mono"
                    >
                      {col}
                    </Badge>
                  ))}
                  {featureCols.length === 0 && (
                    <span className="text-xs text-muted-foreground">
                      Select target first
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 py-3 px-4 rounded-lg bg-secondary/40 border border-border mb-4">
              <Info size={14} className="text-primary shrink-0" />
              <p className="text-xs text-muted-foreground">
                Using Multiple Linear Regression with {featureCols.length}{" "}
                feature{featureCols.length !== 1 ? "s" : ""}. Features are
                normalized before training. OLS via normal equation.
              </p>
            </div>

            <Button
              onClick={trainModel}
              disabled={training || !featureCols.length || !targetCol}
              data-ocid="train.submit_button"
              className="gap-2 w-full sm:w-auto"
            >
              {training ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Training…
                </>
              ) : (
                <>
                  <Brain size={16} /> Train Model
                </>
              )}
            </Button>

            {/* Progress bar */}
            <AnimatePresence>
              {training && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mt-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-muted-foreground">
                      Training…
                    </span>
                    <span className="text-xs font-mono text-primary">
                      {trainProgress}%
                    </span>
                  </div>
                  <Progress
                    value={trainProgress}
                    className="h-2"
                    data-ocid="train.loading_state"
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Metrics */}
          <AnimatePresence>
            {model && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-5"
              >
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatCard
                    label="R² Score"
                    value={model.r2.toFixed(4)}
                    icon={<TrendingUp size={16} />}
                    color="green"
                    sub="Goodness of fit"
                  />
                  <StatCard
                    label="MAE"
                    value={model.mae.toFixed(2)}
                    icon={<Activity size={16} />}
                    color="primary"
                    sub="Mean Abs Error"
                  />
                  <StatCard
                    label="RMSE"
                    value={model.rmse.toFixed(2)}
                    icon={<AlertCircle size={16} />}
                    color="accent"
                    sub="Root Mean Sq Err"
                  />
                  <StatCard
                    label="Samples"
                    value={model.trainSamples}
                    icon={<Database size={16} />}
                    color="purple"
                    sub="Training rows"
                  />
                </div>

                {/* Feature importance */}
                <div className="glass-card rounded-xl p-5">
                  <h3 className="font-semibold text-foreground mb-4">
                    Feature Importance (|Coefficient|)
                  </h3>
                  <ResponsiveContainer
                    width="100%"
                    height={featureImportance.length * 40 + 40}
                  >
                    <BarChart
                      data={featureImportance}
                      layout="vertical"
                      margin={{ left: 10, right: 30 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="oklch(0.22 0.025 258)"
                        horizontal={false}
                      />
                      <XAxis
                        type="number"
                        tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }}
                        width={90}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "oklch(0.13 0.025 258)",
                          border: "1px solid oklch(0.22 0.025 258)",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                        formatter={(val: number) => [
                          val.toFixed(4),
                          "Importance",
                        ]}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {featureImportance.map((entry, i) => (
                          <Cell
                            key={`fi-${entry.feature}`}
                            fill={`oklch(${0.65 + i * 0.04} 0.19 ${197 - i * 10})`}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Coefficients table */}
                <div className="glass-card rounded-xl p-5">
                  <h3 className="font-semibold text-foreground mb-3">
                    Model Coefficients
                  </h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Feature</TableHead>
                        <TableHead>Coefficient</TableHead>
                        <TableHead>Interpretation</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-mono text-xs">
                          (intercept)
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {model.intercept.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          Base value
                        </TableCell>
                      </TableRow>
                      {Object.entries(model.coefficients).map(
                        ([feat, coef], i) => (
                          <TableRow
                            key={feat}
                            data-ocid={`train.coefficients.row.${i + 1}`}
                          >
                            <TableCell className="font-mono text-xs text-primary">
                              {feat}
                            </TableCell>
                            <TableCell
                              className={`font-mono text-xs ${coef >= 0 ? "text-emerald-400" : "text-red-400"}`}
                            >
                              {coef.toFixed(4)}
                            </TableCell>
                            <TableCell className="text-xs text-muted-foreground">
                              +1 unit → {coef >= 0 ? "+" : ""}
                              {coef.toFixed(2)} {targetCol}
                            </TableCell>
                          </TableRow>
                        ),
                      )}
                    </TableBody>
                  </Table>
                </div>

                {/* Model Comparison */}
                <div className="glass-card rounded-xl p-5">
                  <h3 className="font-semibold text-foreground mb-3">
                    Model Comparison
                  </h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Model</TableHead>
                        <TableHead>R²</TableHead>
                        <TableHead>MAE</TableHead>
                        <TableHead>RMSE</TableHead>
                        <TableHead>Best?</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {modelComparison.map((m, i) => {
                        const isBest =
                          i === 0 ||
                          m.r2 ===
                            Math.max(...modelComparison.map((x) => x.r2));
                        return (
                          <TableRow
                            key={m.name}
                            data-ocid={`train.comparison.row.${i + 1}`}
                          >
                            <TableCell className="font-medium text-sm">
                              {m.name}
                            </TableCell>
                            <TableCell className="font-mono text-xs text-emerald-400">
                              {m.r2.toFixed(4)}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {m.mae.toFixed(2)}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {m.rmse.toFixed(2)}
                            </TableCell>
                            <TableCell>
                              {isBest && (
                                <Badge className="bg-emerald-500/20 text-emerald-300 text-xs">
                                  ★ Best
                                </Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                  <ResponsiveContainer
                    width="100%"
                    height={160}
                    className="mt-4"
                  >
                    <BarChart data={modelComparison}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="oklch(0.22 0.025 258)"
                      />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 10 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "oklch(0.13 0.025 258)",
                          border: "1px solid oklch(0.22 0.025 258)",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                        formatter={(val: number) => [val.toFixed(4), "R²"]}
                      />
                      <Bar
                        dataKey="r2"
                        fill="oklch(0.72 0.19 197)"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex justify-end">
                  <Button
                    onClick={() => setActiveSection("predict")}
                    data-ocid="train.predict.button"
                    className="gap-2"
                  >
                    <Zap size={16} /> Run Predictions
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  );

  const renderPredict = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          Predict
        </h2>
        <p className="text-muted-foreground">
          Enter feature values and get instant predictions.
        </p>
      </div>

      {!model ? (
        <EmptyState
          icon={<Zap size={40} />}
          title="No model trained yet"
          action={{
            label: "Train Model",
            onClick: () => setActiveSection("train"),
          }}
        />
      ) : (
        <>
          <div className="glass-card rounded-xl p-5">
            <h3 className="font-semibold text-foreground mb-4">
              Input Features → Predict{" "}
              <span className="text-primary">{targetCol}</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-5">
              {featureCols.map((col, i) => (
                <div key={col}>
                  <Label className="text-xs mb-1.5 block font-mono">
                    {col}
                  </Label>
                  <Input
                    type="number"
                    value={predInputs[col] ?? ""}
                    onChange={(e) =>
                      setPredInputs((prev) => ({
                        ...prev,
                        [col]: e.target.value,
                      }))
                    }
                    data-ocid={`predict.feature.input.${i + 1}`}
                    className="font-mono text-sm h-9"
                    placeholder={`Enter ${col}...`}
                  />
                </div>
              ))}
            </div>

            <Button
              onClick={runPrediction}
              data-ocid="predict.submit_button"
              className="gap-2 w-full sm:w-auto"
            >
              <Zap size={16} /> Calculate Prediction
            </Button>
          </div>

          <AnimatePresence>
            {predResult !== null && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="glass-card rounded-xl p-6"
                data-ocid="predict.success_state"
              >
                <div className="text-center mb-4">
                  <p className="text-sm text-muted-foreground mb-1">
                    Predicted {targetCol}
                  </p>
                  <p className="text-5xl font-bold font-mono text-primary">
                    {predResult.toFixed(2)}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Confidence interval: [{(predResult * 0.9).toFixed(2)},{" "}
                    {(predResult * 1.1).toFixed(2)}]
                  </p>
                </div>

                {/* Input summary */}
                <div className="border border-border rounded-lg p-3 mb-4">
                  <p className="text-xs font-medium text-muted-foreground mb-2">
                    Input features:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {featureCols.map((col) => (
                      <span
                        key={col}
                        className="text-xs font-mono bg-secondary px-2 py-0.5 rounded"
                      >
                        {col}: {predInputs[col]}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button
                    onClick={savePred}
                    disabled={savePrediction.isPending}
                    data-ocid="predict.save_button"
                    variant="outline"
                    className="gap-2"
                  >
                    {savePrediction.isPending ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <History size={14} />
                    )}
                    Save to History
                  </Button>
                  <Button
                    onClick={() => setActiveSection("history")}
                    data-ocid="predict.history.button"
                    variant="ghost"
                    className="gap-2"
                  >
                    View All History
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  );

  const renderHistory = () => (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-display font-bold text-foreground mb-1">
            Prediction History
          </h2>
          <p className="text-muted-foreground">
            All predictions saved to the ICP blockchain.
          </p>
        </div>
        {predictions.length > 0 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => clearPredictions.mutate()}
            disabled={clearPredictions.isPending}
            data-ocid="history.delete_button"
            className="gap-2 shrink-0"
          >
            {clearPredictions.isPending ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Trash2 size={14} />
            )}
            Clear All
          </Button>
        )}
      </div>

      {histLoading ? (
        <div
          className="flex items-center gap-3 text-muted-foreground py-8 justify-center"
          data-ocid="history.loading_state"
        >
          <Loader2 size={20} className="animate-spin" />
          <span>Loading history…</span>
        </div>
      ) : predictions.length === 0 ? (
        <EmptyState
          icon={<History size={40} />}
          title="No predictions yet"
          desc="Run a prediction and save it to see your history here."
          action={{
            label: "Make a Prediction",
            onClick: () => setActiveSection("predict"),
          }}
        />
      ) : (
        <div className="glass-card rounded-xl overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>#</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Input Features</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Timestamp</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.map((p, i) => (
                <TableRow
                  key={`hist-${i}-${p.timestamp}`}
                  data-ocid={`history.item.${i + 1}`}
                >
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {i + 1}
                  </TableCell>
                  <TableCell className="text-xs font-medium">
                    {p.modelName}
                  </TableCell>
                  <TableCell className="text-xs font-mono text-muted-foreground max-w-xs truncate">
                    {p.inputFeatures}
                  </TableCell>
                  <TableCell className="font-mono font-bold text-primary">
                    {p.predictedValue.toFixed(2)}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-xs">
                      {(p.confidence * 100).toFixed(0)}%
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {new Date(Number(p.timestamp)).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* Chart of prediction values */}
      {predictions.length > 1 && (
        <div className="glass-card rounded-xl p-5">
          <h3 className="font-semibold text-foreground mb-4">
            Prediction Trend
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={predictions.map((p, i) => ({
                index: i + 1,
                value: p.predictedValue,
              }))}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="oklch(0.22 0.025 258)"
              />
              <XAxis
                dataKey="index"
                tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }}
              />
              <YAxis tick={{ fill: "oklch(0.55 0.02 255)", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "oklch(0.13 0.025 258)",
                  border: "1px solid oklch(0.22 0.025 258)",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="oklch(0.72 0.19 197)"
                strokeWidth={2}
                dot={{ fill: "oklch(0.72 0.19 197)", r: 4 }}
                name="Predicted Value"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );

  const renderGuide = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-1">
          User Guide
        </h2>
        <p className="text-muted-foreground">
          Learn how to use ML Studio effectively.
        </p>
      </div>

      {[
        {
          icon: <Upload size={20} />,
          title: "1. Upload Your Dataset",
          content:
            "Go to Upload Data and drag-and-drop any CSV file. The first row must contain column headers. All numeric columns will be auto-detected. You can also use the demo housing dataset to explore features without uploading your own data.",
        },
        {
          icon: <Search size={20} />,
          title: "2. Explore Your Data",
          content:
            "In the Explore section, review dataset statistics (min, max, mean, std dev) for each column, preview the first 10 rows, and select which column you want to predict (the target). All remaining numeric columns become features.",
        },
        {
          icon: <BarChart2 size={20} />,
          title: "3. Visualize Relationships",
          content:
            "Use Visualize to create scatter plots between any two columns, view the distribution histogram of any column, and check Pearson correlation coefficients between each feature and the target column.",
        },
        {
          icon: <Brain size={20} />,
          title: "4. Train a Model",
          content:
            "In Train Model, confirm your target column and click Train. The app fits a Multiple Linear Regression using all numeric features. Model metrics (R², MAE, RMSE) are shown, along with a feature importance chart and a model comparison table (Linear, Ridge α=1, Ridge α=10).",
        },
        {
          icon: <Zap size={20} />,
          title: "5. Make Predictions",
          content:
            "In Predict, enter values for each feature. Click Calculate to get the predicted value with a ±10% confidence interval. Save predictions to the ICP backend using Save to History.",
        },
        {
          icon: <History size={20} />,
          title: "6. Review History",
          content:
            "All saved predictions are stored on the Internet Computer blockchain. The History section shows every prediction with inputs, model name, timestamp, and confidence. You can clear the history at any time.",
        },
      ].map((item) => (
        <div key={item.title} className="glass-card rounded-xl p-5 flex gap-4">
          <div className="p-2 rounded-lg bg-primary/10 text-primary h-fit shrink-0">
            {item.icon}
          </div>
          <div>
            <h3 className="font-semibold text-foreground mb-1.5">
              {item.title}
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {item.content}
            </p>
          </div>
        </div>
      ))}

      <div className="glass-card rounded-xl p-5 border border-primary/20">
        <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
          <Info size={16} className="text-primary" />
          Tips & Best Practices
        </h3>
        <ul className="space-y-2">
          {[
            "Datasets with 50+ rows produce more reliable models.",
            "Highly correlated features can cause coefficient instability — use Ridge regression in those cases.",
            "R² close to 1.0 means the model explains most variance. Below 0.5 suggests weak predictive power.",
            "RMSE is in the same units as your target column, making it easy to interpret.",
            "If predictions seem unreasonable, check for outliers in your dataset.",
          ].map((tip) => (
            <li
              key={tip}
              className="flex items-start gap-2 text-sm text-muted-foreground"
            >
              <span className="text-primary font-bold shrink-0">→</span>
              {tip}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  const sectionRenderers: Record<Section, () => React.ReactNode> = {
    dashboard: renderDashboard,
    upload: renderUpload,
    explore: renderExplore,
    visualize: renderVisualize,
    train: renderTrain,
    predict: renderPredict,
    history: renderHistory,
    guide: renderGuide,
  };

  // ─── Layout Render ────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Toaster richColors position="top-right" />

      {/* ── Sidebar overlay (mobile) ── */}
      <AnimatePresence>
        {sidebarOpen && isMobile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 z-30 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* ── Sidebar ── */}
      <motion.aside
        animate={{ width: sidebarOpen ? 240 : 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className={`relative flex-shrink-0 overflow-hidden ${
          isMobile ? "fixed top-0 left-0 h-full z-40" : "relative z-20"
        }`}
        style={{ minWidth: 0 }}
      >
        <div
          className="w-60 h-full flex flex-col bg-sidebar border-r border-sidebar-border"
          style={{ width: 240 }}
        >
          {/* Logo */}
          <div className="p-4 flex items-center gap-3 border-b border-sidebar-border">
            <div className="p-2 rounded-lg bg-primary/15">
              <FlaskConical size={20} className="text-primary" />
            </div>
            <div>
              <p className="font-display font-bold text-sm text-foreground">
                ML Studio
              </p>
              <p className="text-xs text-muted-foreground">v2.0 Professional</p>
            </div>
            {isMobile && (
              <button
                type="button"
                onClick={() => setSidebarOpen(false)}
                className="ml-auto p-1 rounded hover:bg-secondary text-muted-foreground"
                data-ocid="nav.close_button"
              >
                <X size={16} />
              </button>
            )}
          </div>

          {/* Nav */}
          <ScrollArea className="flex-1 py-3">
            <nav className="px-2 space-y-0.5">
              {NAV_ITEMS.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => navigate(item.id)}
                  data-ocid={`nav.${item.id}.link`}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                    activeSection === item.id
                      ? "bg-primary/15 text-primary"
                      : "text-sidebar-foreground hover:bg-secondary/60 hover:text-foreground"
                  }`}
                >
                  <span className="shrink-0">{item.icon}</span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{item.label}</p>
                    <p className="text-xs text-muted-foreground truncate">
                      {item.desc}
                    </p>
                  </div>
                  {activeSection === item.id && (
                    <span className="ml-auto w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
                  )}
                </button>
              ))}
            </nav>

            <Separator className="my-3 mx-2" />

            {/* Dataset status in sidebar */}
            <div className="px-3 py-2">
              <p className="text-xs font-medium text-muted-foreground mb-2">
                Dataset
              </p>
              {data.length > 0 ? (
                <div className="text-xs">
                  <p
                    className="font-mono text-foreground truncate"
                    title={datasetName}
                  >
                    {datasetName}
                  </p>
                  <p className="text-muted-foreground">
                    {data.length} rows · {Object.keys(data[0]).length} cols
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No data loaded</p>
              )}
            </div>
          </ScrollArea>

          {/* Footer */}
          <div className="p-3 border-t border-sidebar-border">
            <p className="text-xs text-muted-foreground text-center">
              © {new Date().getFullYear()}.{" "}
              <a
                href={`https://caffeine.ai?utm_source=caffeine-footer&utm_medium=referral&utm_content=${encodeURIComponent(window.location.hostname)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary transition-colors"
              >
                caffeine.ai
              </a>
            </p>
          </div>
        </div>
      </motion.aside>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-border bg-card/50 backdrop-blur-sm shrink-0">
          <button
            type="button"
            onClick={() => setSidebarOpen((v) => !v)}
            data-ocid="nav.toggle"
            className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            {sidebarOpen && !isMobile ? (
              <ChevronLeft size={18} />
            ) : (
              <ChevronRight size={18} />
            )}
          </button>

          <div className="flex items-center gap-2 min-w-0">
            <h1 className="font-display font-bold text-foreground text-sm truncate">
              {NAV_ITEMS.find((n) => n.id === activeSection)?.label}
            </h1>
          </div>

          <div className="ml-auto flex items-center gap-2 flex-wrap">
            {data.length > 0 && (
              <>
                <Badge
                  variant="outline"
                  className="font-mono text-xs gap-1 hidden sm:flex"
                >
                  <FileText size={10} />
                  {datasetName}
                </Badge>
                <Badge
                  variant="secondary"
                  className="font-mono text-xs hidden sm:flex"
                >
                  {data.length}×{Object.keys(data[0]).length}
                </Badge>
              </>
            )}
            <Badge className={statusBadge.className}>
              {status === "trained" ? (
                <CheckCircle2 size={10} className="mr-1" />
              ) : status === "loaded" ? (
                <Database size={10} className="mr-1" />
              ) : null}
              {statusBadge.label}
            </Badge>
            {model && (
              <button
                type="button"
                onClick={() => {
                  setModel(null);
                  setPredResult(null);
                  toast.info("Model cleared");
                }}
                data-ocid="header.reset.button"
                className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                title="Clear model"
              >
                <RefreshCw size={14} />
              </button>
            )}
          </div>
        </header>

        {/* Content */}
        <ScrollArea className="flex-1">
          <main className="p-4 md:p-6 max-w-5xl mx-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
              >
                {sectionRenderers[activeSection]()}
              </motion.div>
            </AnimatePresence>
          </main>
        </ScrollArea>
      </div>
    </div>
  );
}

// ─── Empty State Component ────────────────────────────────────────────────────
function EmptyState({
  icon,
  title,
  desc,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  desc?: string;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <div
      className="glass-card rounded-xl p-12 flex flex-col items-center gap-4 text-center"
      data-ocid="content.empty_state"
    >
      <div className="p-4 rounded-full bg-secondary text-muted-foreground">
        {icon}
      </div>
      <div>
        <p className="font-semibold text-foreground">{title}</p>
        {desc && <p className="text-sm text-muted-foreground mt-1">{desc}</p>}
      </div>
      {action && (
        <Button
          variant="outline"
          onClick={action.onClick}
          data-ocid="content.empty_state.button"
        >
          {action.label}
        </Button>
      )}
    </div>
  );
}
