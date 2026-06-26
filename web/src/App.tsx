import { useState } from "react";
import { motion } from "framer-motion";
import { Play, Loader2, LineChart, Activity, FlaskConical } from "lucide-react";

type Hist = {
  loss: number[];
  val_acc: number[];
  final_val_acc: number;
  min_loss: number;
  epochs: number;
};

const COLORS = ["#22d3ee", "#a78bfa", "#fb923c", "#f43f5e", "#34d399"];

function Chart({
  series,
  yMax,
  yMin = 0,
  height = 220,
  label,
}: {
  series: { name: string; color: string; points: number[] }[];
  yMax: number;
  yMin?: number;
  height?: number;
  label: string;
}) {
  const W = 560;
  const H = height;
  const pad = 34;
  const maxLen = Math.max(1, ...series.map((s) => s.points.length));
  const x = (i: number) =>
    pad + (i / Math.max(1, maxLen - 1)) * (W - pad * 1.5);
  const y = (v: number) =>
    H - pad - ((v - yMin) / (yMax - yMin || 1)) * (H - pad * 1.8);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={pad}
            x2={W - pad * 0.5}
            y1={H - pad - t * (H - pad * 1.8)}
            y2={H - pad - t * (H - pad * 1.8)}
            stroke="rgba(255,255,255,0.06)"
          />
        ))}
        {series.map((s) => (
          <motion.polyline
            key={s.name}
            fill="none"
            stroke={s.color}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            points={s.points.map((p, i) => `${x(i)},${y(p)}`).join(" ")}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.8 }}
          />
        ))}
        <text x={pad} y={H - 8} fill="rgba(255,255,255,0.4)" fontSize="11">epoch</text>
      </svg>
      {series.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-3">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className="h-2 w-3 rounded-full" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Blobs() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="absolute -top-44 -left-40 h-[34rem] w-[34rem] rounded-full bg-orange-600/15 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-[30rem] w-[30rem] rounded-full bg-violet-500/15 blur-3xl animate-float [animation-delay:-6s]" />
    </div>
  );
}

const LR_PRESETS = [
  { label: "too low", lr: 0.003 },
  { label: "good", lr: 0.1 },
  { label: "too high", lr: 0.5 },
  { label: "diverges", lr: 3.0 },
];

export default function App() {
  const [optimizer, setOptimizer] = useState("sgd_momentum");
  const [lr, setLr] = useState(0.1);
  const [epochs, setEpochs] = useState(30);
  const [hist, setHist] = useState<Hist | null>(null);
  const [ablation, setAblation] = useState<Record<string, Hist> | null>(null);
  const [loading, setLoading] = useState<"train" | "ablation" | null>(null);

  async function train() {
    setLoading("train");
    setAblation(null);
    try {
      const r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ optimizer, lr, epochs }),
      });
      setHist(await r.json());
    } finally {
      setLoading(null);
    }
  }

  async function runAblation() {
    setLoading("ablation");
    setHist(null);
    try {
      const r = await fetch("/api/ablation");
      setAblation((await r.json()).runs);
    } finally {
      setLoading(null);
    }
  }

  const lossMax = hist ? Math.max(...hist.loss) : 1;

  return (
    <div className="relative min-h-screen text-slate-200">
      <Blobs />
      <div className="relative mx-auto max-w-3xl px-5 py-14">
        <motion.header
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs text-slate-300 backdrop-blur">
            <Activity size={14} className="text-orange-300" />
            Train a neural net live · read the curve
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-6xl">
            Training{" "}
            <span className="bg-gradient-to-r from-orange-400 via-amber-300 to-violet-300 bg-clip-text text-transparent">
              Dashboard
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-lg text-slate-400">
            Pick the optimizer and learning rate, train an MLP on digits, and watch
            the loss curve — or run the ablation to see a too-high LR diverge.
          </p>
        </motion.header>

        {/* Controls */}
        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5 backdrop-blur">
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="text-sm">
              <span className="mb-1 block text-slate-400">Optimizer</span>
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-slate-200 outline-none"
              >
                <option value="sgd_momentum">SGD + momentum</option>
                <option value="sgd">SGD</option>
                <option value="adam">Adam</option>
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-slate-400">Learning rate · {lr}</span>
              <input
                type="number"
                step="0.001"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value) || 0)}
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-slate-200 outline-none"
              />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-slate-400">Epochs · {epochs}</span>
              <input
                type="range"
                min={5}
                max={60}
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full accent-orange-400"
              />
            </label>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500">LR presets:</span>
            {LR_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setLr(p.lr)}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400 hover:bg-white/10"
              >
                {p.label} ({p.lr})
              </button>
            ))}
            <div className="ml-auto flex gap-2">
              <button
                onClick={runAblation}
                disabled={loading !== null}
                className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-white/10 disabled:opacity-60"
              >
                {loading === "ablation" ? <Loader2 className="animate-spin" size={15} /> : <FlaskConical size={15} />}
                Run ablation
              </button>
              <button
                onClick={train}
                disabled={loading !== null}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 px-5 py-2 text-sm font-semibold text-white disabled:opacity-60"
              >
                {loading === "train" ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                Train
              </button>
            </div>
          </div>
        </div>

        {/* Single run */}
        {hist && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mt-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-center backdrop-blur">
                <div className="text-3xl font-extrabold text-emerald-300">
                  {(hist.final_val_acc * 100).toFixed(1)}%
                </div>
                <div className="text-[11px] uppercase tracking-wider text-slate-500">final val accuracy</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-center backdrop-blur">
                <div className="font-mono text-3xl font-extrabold text-orange-300">{hist.min_loss}</div>
                <div className="text-[11px] uppercase tracking-wider text-slate-500">min train loss</div>
              </div>
            </div>
            <Chart label="Training loss" series={[{ name: "loss", color: "#fb923c", points: hist.loss }]} yMax={lossMax} />
            <Chart label="Validation accuracy" series={[{ name: "val_acc", color: "#34d399", points: hist.val_acc }]} yMax={1} />
          </motion.div>
        )}

        {/* Ablation */}
        {ablation && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mt-6 space-y-4">
            <Chart
              label="Training loss — learning-rate ablation"
              yMax={Math.max(...Object.values(ablation).flatMap((h) => h.loss))}
              series={Object.entries(ablation).map(([name, h], i) => ({
                name,
                color: COLORS[i % COLORS.length],
                points: h.loss,
              }))}
            />
            <Chart
              label="Validation accuracy — learning-rate ablation"
              yMax={1}
              series={Object.entries(ablation).map(([name, h], i) => ({
                name,
                color: COLORS[i % COLORS.length],
                points: h.val_acc,
              }))}
            />
            <p className="text-center text-sm text-slate-500">
              <LineChart className="mr-1 inline" size={14} />
              Watch the too-high learning rates flatline near 10% (random) — the
              signature of divergence.
            </p>
          </motion.div>
        )}

        <footer className="mt-16 text-center text-xs text-slate-600">
          Small MLP on sklearn digits · trains in seconds · TensorBoard tracking in
          the CLI version
        </footer>
      </div>
    </div>
  );
}
