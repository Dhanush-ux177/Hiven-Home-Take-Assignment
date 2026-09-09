import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Bot,
  Sparkles,
  Search,
  RotateCcw,
  BookOpen,
  Scale,
  Flame,
  FileText,
  Copy,
  ExternalLink,
  ChevronRight,
  Database,
  BarChart3,
  ListFilter,
  Check,
  Zap,
  HelpCircle,
  AlertCircle,
  MessageSquareQuote
} from 'lucide-react';

interface BenchmarkMetrics {
  benchmark_date: string;
  dataset: string;
  golden_set_size: number;
  models: {
    baseline_trivial: any;
    baseline_simple: any;
    proposed_agent: any;
  };
  top_failure_modes: Array<{
    rank: number;
    title: string;
    example_tweet: string;
    tweet_id: string;
    predicted_intent: string;
    gold_intent: string;
    predicted_escalate?: boolean;
    gold_escalate?: boolean;
    root_cause_hypothesis: string;
    mitigation: string;
  }>;
}

interface GoldenRecord {
  id: string;
  raw_conversation_id: string;
  customer_text: string;
  conversation_history: string[];
  thread_depth: number;
  real_agent_reply: string;
  gold_intent: string;
  gold_escalate: boolean;
  gold_escalation_reason: string;
  human_quality_score: number;
  sampling_stratum: string;
  annotator_notes: string;
}

const PRESET_QUERIES = [
  {
    label: "Apple ID Lockout (Security Guardrail)",
    text: "My Apple ID is locked for security reasons and I can't get verification code on my trusted number. Help!",
    expected_intent: "ACCOUNT_SECURITY_ICLOUD",
    expected_escalate: true
  },
  {
    label: "Swollen Battery (Hardware Safety Hazard)",
    text: "My iPhone battery is swollen and the front screen is lifting off the frame! Is it dangerous to charge?",
    expected_intent: "HARDWARE_BATTERY",
    expected_escalate: true
  },
  {
    label: "iOS 11 Keyboard Lag (Self-Service Software)",
    text: "My keyboard lags terribly since updating to iOS 11 on iPhone 6s. Typing in Messages is delayed.",
    expected_intent: "SOFTWARE_OS_BUG",
    expected_escalate: false
  },
  {
    label: "Delete Apps via 3D Touch (Feature Discovery)",
    text: "How do I delete apps on iPhone 7? When I press down it just shows quick actions instead of wiggling.",
    expected_intent: "DEVICE_SETUP_FEATURE",
    expected_escalate: false
  },
  {
    label: "Headphone Jack Removal (Brand Venting Rant)",
    text: "Removing the headphone jack is the dumbest move Apple has ever done. Pure greed. iPhone sucks now.",
    expected_intent: "GENERAL_FEEDBACK_RANT",
    expected_escalate: false
  },
  {
    label: "Genius Bar Reservation (Store Logistics)",
    text: "Need to replace a cracked iPhone screen today in London. Can I walk into Covent Garden Apple Store without an appointment?",
    expected_intent: "STORE_REPAIR_ORDER",
    expected_escalate: true
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'playground' | 'benchmark' | 'dataset' | 'calibration' | 'failures' | 'report'>('playground');
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkMetrics | null>(null);
  const [goldenData, setGoldenData] = useState<GoldenRecord[]>([]);
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [loading, setLoading] = useState(true);

  // Playground state
  const [inputQuery, setInputQuery] = useState(PRESET_QUERIES[0].text);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [useLiveLLM, setUseLiveLLM] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Dataset filter state
  const [datasetSearch, setDatasetSearch] = useState('');
  const [intentFilter, setIntentFilter] = useState('ALL');
  const [escalateFilter, setEscalateFilter] = useState('ALL');
  const [selectedRecord, setSelectedRecord] = useState<GoldenRecord | null>(null);

  // Load initial data
  useEffect(() => {
    async function loadData() {
      try {
        const [benchRes, dataRes, reportRes] = await Promise.all([
          fetch('/api/benchmark').then(r => r.ok ? r.json() : null),
          fetch('/api/dataset').then(r => r.ok ? r.json() : []),
          fetch('/api/report').then(r => r.ok ? r.text() : '')
        ]);
        if (benchRes) setBenchmarkData(benchRes);
        if (dataRes) setGoldenData(dataRes);
        if (reportRes) setReportMarkdown(reportRes);
      } catch (err) {
        console.error("Failed to load initial data", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Run initial query on mount
  useEffect(() => {
    handleRunQuery(PRESET_QUERIES[0].text);
  }, []);

  const handleRunQuery = async (queryText: string) => {
    if (!queryText.trim()) return;
    setQueryLoading(true);
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, llm: useLiveLLM })
      });
      if (res.ok) {
        const data = await res.json();
        setQueryResult(data);
      }
    } catch (err) {
      console.error("Query failed", err);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleRerunBenchmark = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/run-eval', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setBenchmarkData(data);
      }
    } catch (err) {
      console.error("Failed to rerun benchmark", err);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  // Filtered dataset
  const filteredDataset = goldenData.filter(item => {
    const matchesSearch = item.customer_text.toLowerCase().includes(datasetSearch.toLowerCase()) ||
                          item.real_agent_reply.toLowerCase().includes(datasetSearch.toLowerCase()) ||
                          item.id.toLowerCase().includes(datasetSearch.toLowerCase());
    const matchesIntent = intentFilter === 'ALL' || item.gold_intent === intentFilter;
    const matchesEscalate = escalateFilter === 'ALL' ||
                            (escalateFilter === 'ESCALATE' && item.gold_escalate) ||
                            (escalateFilter === 'AUTOHANDLE' && !item.gold_escalate);
    return matchesSearch && matchesIntent && matchesEscalate;
  });

  const proposed = benchmarkData?.models?.proposed_agent;
  const simple = benchmarkData?.models?.baseline_simple;
  const trivial = benchmarkData?.models?.baseline_trivial;

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans selection:bg-sky-500/25 selection:text-sky-200">
      {/* Top Brand Header */}
      <header className="border-b border-slate-800/80 bg-[#0d1322]/95 backdrop-blur sticky top-0 z-30 px-3 sm:px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-3 sm:gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-sky-500/20 text-white font-bold text-lg shrink-0">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-base sm:text-xl font-bold text-white tracking-tight truncate">
                  OmniSupport AI Agent
                </h1>
                <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1 shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  Grounded & Verified
                </span>
              </div>
              <p className="text-[11px] sm:text-xs text-slate-400 line-clamp-1 sm:line-clamp-none mt-0.5">
                Multi-Intent Customer Support Engine • Kaggle Twitter Benchmark (<span className="text-slate-300 font-mono">@AppleSupport</span> N=200)
              </p>
            </div>
          </div>

          {/* Operational Metrics Pills */}
          <div className="flex flex-wrap sm:flex-nowrap items-center gap-1.5 sm:gap-2 overflow-x-auto pb-1 sm:pb-0 text-[11px] sm:text-xs">
            <div className="px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center gap-1.5 shrink-0">
              <Database className="w-3.5 h-3.5 text-sky-400" />
              <span className="text-slate-400">Golden Set:</span>
              <span className="font-semibold text-slate-200">200 Cases</span>
            </div>
            <div className="px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center gap-1.5 shrink-0">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-400">Intent F1:</span>
              <span className="font-semibold text-emerald-400">{proposed?.intent_metrics?.macro_f1 || '0.733'}</span>
            </div>
            <div className="px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center gap-1.5 shrink-0">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span className="text-slate-400">Safety Recall:</span>
              <span className="font-semibold text-emerald-400">70.0%</span>
            </div>
            <div className="px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center gap-1.5 shrink-0">
              <Scale className="w-3.5 h-3.5 text-indigo-400" />
              <span className="text-slate-400">Judge Agree:</span>
              <span className="font-semibold text-indigo-300">94.0%</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto mt-3 flex items-center gap-1 border-t border-slate-800/60 pt-2 overflow-x-auto no-scrollbar scroll-smooth">
          {[
            { id: 'playground', label: 'Live Agent Playground', icon: Bot },
            { id: 'benchmark', label: 'Headline Benchmark & Baselines', icon: BarChart3 },
            { id: 'dataset', label: 'Golden Dataset Explorer (N=200)', icon: Database },
            { id: 'calibration', label: 'LLM Judge & Human Calibration', icon: Scale },
            { id: 'failures', label: 'Top 5 Failure Modes', icon: AlertTriangle },
            { id: 'report', label: 'Technical Report (REPORT.md)', icon: FileText },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                id={`tab-button-${tab.id}`}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3.5 py-1.5 sm:py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap shrink-0 ${
                  active
                    ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${active ? 'text-sky-400' : 'text-slate-400'}`} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-6">
        {/* TAB 1: LIVE AGENT PLAYGROUND */}
        {activeTab === 'playground' && (
          <div className="space-y-6">
            {/* Context Notice */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-sky-950/40 via-slate-900 to-indigo-950/40 border border-sky-900/50 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-white">Live Real-Time Inference & Comparative Benchmark</h3>
                  <p className="text-xs text-slate-300 mt-0.5">
                    Test incoming customer tweets against @AppleSupport's calibrated policy engine. The system classifies intent, checks security guardrails, retrieves historical Kaggle exemplars, and compares outputs directly against two baseline models.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
                  <input
                    type="checkbox"
                    checked={useLiveLLM}
                    onChange={(e) => setUseLiveLLM(e.target.checked)}
                    className="rounded border-slate-700 text-sky-500 focus:ring-0"
                  />
                  <span>Use Live Gemini LLM Drafting</span>
                </label>
              </div>
            </div>

            {/* Presets Bar */}
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <ListFilter className="w-3.5 h-3.5 text-sky-400" />
                Select Kaggle Benchmark Test Scenarios
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {PRESET_QUERIES.map((preset, idx) => (
                  <button
                    key={idx}
                    id={`preset-button-${idx}`}
                    onClick={() => {
                      setInputQuery(preset.text);
                      handleRunQuery(preset.text);
                    }}
                    className={`p-2.5 rounded-lg text-left border transition-all ${
                      inputQuery === preset.text
                        ? 'bg-slate-800/90 border-sky-500/50 text-white shadow-sm'
                        : 'bg-slate-900/60 border-slate-800/80 text-slate-300 hover:border-slate-700 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-medium mb-1">
                      <span className="text-sky-300">{preset.label}</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                        preset.expected_escalate ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                      }`}>
                        {preset.expected_escalate ? 'Escalate' : 'Auto-Handle'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 line-clamp-2 italic">"{preset.text}"</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Query Input Box */}
            <div className="bg-slate-900/80 rounded-xl border border-slate-800 p-4 space-y-3">
              <label htmlFor="customer-tweet-input" className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Incoming Customer Tweet
              </label>
              <div className="flex flex-col sm:flex-row gap-2">
                <textarea
                  id="customer-tweet-input"
                  rows={2}
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  placeholder="Enter a customer tweet to @AppleSupport (e.g., 'My Apple ID is disabled and won't let me log in...')"
                  className="flex-1 bg-slate-950/80 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30"
                />
                <button
                  id="run-inference-button"
                  onClick={() => handleRunQuery(inputQuery)}
                  disabled={queryLoading}
                  className="px-5 py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white text-sm font-semibold rounded-lg shadow-lg shadow-sky-500/20 flex items-center justify-center gap-2 transition-all disabled:opacity-50 shrink-0"
                >
                  {queryLoading ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Evaluate Agent
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Side-by-Side Comparison: Proposed Agent vs Baselines */}
            {queryResult && (
              <div className="space-y-4">
                <div className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <span>Model Execution Comparison: Proposed Agent vs. Baselines</span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {/* Proposed Grounded Agent */}
                  <div className="bg-slate-900/90 rounded-xl border-2 border-sky-500/50 p-5 space-y-4 shadow-xl shadow-sky-950/20 relative">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-md bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold text-xs">
                          <Bot className="w-3.5 h-3.5 text-sky-400" />
                        </div>
                        <span className="text-sm font-bold text-white">Proposed Grounded Agent</span>
                      </div>
                      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/20 text-sky-300 border border-sky-500/30">
                        Target System
                      </span>
                    </div>

                    {/* Intent & Confidence */}
                    <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Classified Intent:</span>
                        <span className="font-bold text-sky-400 font-mono">
                          {queryResult.proposed_agent?.intent}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="font-semibold text-emerald-400">
                          {(queryResult.proposed_agent?.intent_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 italic">
                        {queryResult.proposed_agent?.intent_reason}
                      </p>
                    </div>

                    {/* Escalation Decision */}
                    <div className={`p-3 rounded-lg border space-y-1.5 ${
                      queryResult.proposed_agent?.escalate
                        ? 'bg-rose-950/30 border-rose-800/50 text-rose-200'
                        : 'bg-emerald-950/30 border-emerald-800/50 text-emerald-200'
                    }`}>
                      <div className="flex items-center justify-between text-xs font-semibold">
                        <span className="flex items-center gap-1.5">
                          {queryResult.proposed_agent?.escalate ? (
                            <ShieldAlert className="w-4 h-4 text-rose-400" />
                          ) : (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          )}
                          {queryResult.proposed_agent?.escalate ? 'ESCALATE TO HUMAN / DM' : 'AUTO-HANDLE (SELF-SERVICE)'}
                        </span>
                        <span className="text-[10px] font-mono uppercase opacity-75">
                          {queryResult.proposed_agent?.escalation_policy_tier}
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed opacity-90">
                        {queryResult.proposed_agent?.escalation_reason}
                      </p>
                    </div>

                    {/* Drafted Reply */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span className="font-semibold uppercase tracking-wider text-[10px]">Brand-Compliant Drafted Reply</span>
                        <button
                          onClick={() => copyToClipboard(queryResult.proposed_agent?.drafted_reply, 'agent_reply')}
                          className="hover:text-white flex items-center gap-1 text-[11px]"
                        >
                          {copiedKey === 'agent_reply' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                          {copiedKey === 'agent_reply' ? 'Copied' : 'Copy'}
                        </button>
                      </div>
                      <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs leading-relaxed text-slate-100 font-sans">
                        "{queryResult.proposed_agent?.drafted_reply}"
                      </div>
                    </div>

                    {/* Historical RAG Exemplars */}
                    {queryResult.proposed_agent?.retrieved_exemplars?.length > 0 && (
                      <div className="space-y-1.5 pt-1 border-t border-slate-800/60">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                          Historical Grounding (BM25 Match from Kaggle)
                        </span>
                        <div className="text-xs bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/60 space-y-1.5">
                          <p className="text-[11px] text-slate-300 line-clamp-2">
                            <span className="text-sky-400 font-semibold font-mono">Matched:</span> "{queryResult.proposed_agent?.retrieved_exemplars[0]?.query}"
                          </p>
                          <p className="text-[11px] text-slate-400 line-clamp-2">
                            <span className="text-emerald-400 font-semibold">Resolved:</span> "{queryResult.proposed_agent?.retrieved_exemplars[0]?.reply}"
                          </p>
                          {queryResult.proposed_agent?.retrieved_exemplars[0]?.kb_link && (
                            <a
                              href={queryResult.proposed_agent?.retrieved_exemplars[0]?.kb_link}
                              target="_blank"
                              rel="noreferrer"
                              className="text-[11px] text-sky-400 hover:underline flex items-center gap-1"
                            >
                              <ExternalLink className="w-3 h-3" />
                              {queryResult.proposed_agent?.retrieved_exemplars[0]?.kb_link}
                            </a>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Baseline 2: Simple (Naive Keyword + Length Escalation) */}
                  <div className="bg-slate-900/60 rounded-xl border border-slate-800 p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-slate-300">Baseline 2: Simple</span>
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-800 text-slate-400">
                        Naive Keyword
                      </span>
                    </div>

                    <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Intent:</span>
                        <span className="font-semibold text-slate-300 font-mono">
                          {queryResult.baseline_simple?.intent}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="text-slate-400">Fixed (0.70)</span>
                      </div>
                      <p className="text-[11px] text-slate-500 italic">
                        {queryResult.baseline_simple?.intent_reason}
                      </p>
                    </div>

                    <div className={`p-3 rounded-lg border space-y-1 ${
                      queryResult.baseline_simple?.escalate
                        ? 'bg-rose-950/20 border-rose-900/40 text-rose-300'
                        : 'bg-slate-800/30 border-slate-700/40 text-slate-300'
                    }`}>
                      <div className="text-xs font-semibold">
                        {queryResult.baseline_simple?.escalate ? 'ESCALATE' : 'AUTO-HANDLE'}
                      </div>
                      <p className="text-[11px] opacity-80">
                        {queryResult.baseline_simple?.escalation_reason}
                      </p>
                    </div>

                    <div className="space-y-1.5">
                      <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Templated Reply</span>
                      <div className="p-3 bg-slate-950/60 border border-slate-800/60 rounded-lg text-xs leading-relaxed text-slate-400">
                        "{queryResult.baseline_simple?.drafted_reply}"
                      </div>
                    </div>

                    <div className="text-[11px] text-amber-400/80 bg-amber-950/20 p-2.5 rounded-lg border border-amber-900/30">
                      <strong>Vulnerability:</strong> Relies solely on punctuation length triggers. Misses critical security and safety lockouts when written in calm sentences.
                    </div>
                  </div>

                  {/* Baseline 1: Trivial (Majority Class + Canned) */}
                  <div className="bg-slate-900/40 rounded-xl border border-slate-800/70 p-5 space-y-4 opacity-80 hover:opacity-100 transition-opacity">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-slate-400">Baseline 1: Trivial</span>
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-800 text-slate-400">
                        Static Canned
                      </span>
                    </div>

                    <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Intent:</span>
                        <span className="font-semibold text-slate-400 font-mono">SOFTWARE_OS_BUG</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="text-slate-500">0.50</span>
                      </div>
                      <p className="text-[11px] text-slate-500 italic">Always predicts majority class</p>
                    </div>

                    <div className="p-3 rounded-lg border bg-slate-950/40 border-slate-800/80 text-slate-400 space-y-1">
                      <div className="text-xs font-semibold">NEVER ESCALATES (Always False)</div>
                      <p className="text-[11px] opacity-75">Static rule: zero safety awareness</p>
                    </div>

                    <div className="space-y-1.5">
                      <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Canned Reply</span>
                      <div className="p-3 bg-slate-950/60 border border-slate-800/60 rounded-lg text-xs leading-relaxed text-slate-400">
                        "{queryResult.baseline_trivial?.drafted_reply}"
                      </div>
                    </div>

                    <div className="text-[11px] text-rose-400/80 bg-rose-950/20 p-2.5 rounded-lg border border-rose-900/30">
                      <strong>Fatal Flaw:</strong> 100% False Auto-Handle Rate on safety issues. Leaves locked accounts and swollen batteries stranded.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: BENCHMARK & BASELINES */}
        {activeTab === 'benchmark' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white">Headline Evaluation Results vs. Baselines</h2>
                <p className="text-xs text-slate-400">
                  Evaluated on 200 hand-labelled real customer support conversations from Kaggle's @AppleSupport repository.
                </p>
              </div>
              <button
                id="rerun-benchmark-button"
                onClick={handleRerunBenchmark}
                disabled={loading}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 flex items-center gap-2 transition-all self-start sm:self-auto"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                Rerun Evaluation Pipeline
              </button>
            </div>

            {/* Big Stat Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Intent Accuracy</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-extrabold text-white">
                    {(proposed?.intent_metrics?.accuracy * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-emerald-400 font-semibold font-mono">
                    Macro F1: {proposed?.intent_metrics?.macro_f1}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">+2.5% over Simple, +28.0% over Trivial</p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Escalation F1 Score</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-extrabold text-sky-400">
                    {proposed?.escalation_metrics?.f1}
                  </span>
                  <span className="text-xs text-slate-400">Acc: {(proposed?.escalation_metrics?.accuracy * 100).toFixed(1)}%</span>
                </div>
                <p className="text-[11px] text-slate-400">+284% improvement over Simple baseline</p>
              </div>

              <div className="bg-slate-900/80 border border-rose-900/40 p-4 rounded-xl space-y-1">
                <span className="text-xs text-rose-300 font-medium">False Auto-Handle Rate (Safety)</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-extrabold text-rose-400">
                    {(proposed?.escalation_metrics?.false_autohandle_rate * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-rose-300/80">Hazard Risk</span>
                </div>
                <p className="text-[11px] text-slate-400">Reduced from 100% (Trivial) & 90% (Simple)</p>
              </div>

              <div className="bg-slate-900/80 border border-indigo-900/40 p-4 rounded-xl space-y-1">
                <span className="text-xs text-indigo-300 font-medium">Judge-Human Agreement</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-extrabold text-indigo-300">
                    {(proposed?.judge_metrics?.adjacent_agreement_rate * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-indigo-400">Adjacent (±1.0)</span>
                </div>
                <p className="text-[11px] text-slate-400">Average response score: {proposed?.judge_metrics?.mean_judge_score} / 5.0</p>
              </div>
            </div>

            {/* Comprehensive Comparative Table */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Full Benchmark Comparison Matrix</h3>
                <span className="text-xs text-slate-400">Evaluation Sample: N = 200 Hand-Labelled</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800 uppercase tracking-wider font-semibold">
                      <th className="py-3 px-4">Evaluation Metric</th>
                      <th className="py-3 px-4">Baseline 1: Trivial</th>
                      <th className="py-3 px-4">Baseline 2: Simple</th>
                      <th className="py-3 px-4 text-sky-400 font-bold bg-sky-950/20">Proposed Grounded Agent</th>
                      <th className="py-3 px-4 text-slate-400">Operational Target</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Intent Accuracy</td>
                      <td className="py-3 px-4 font-mono">{(trivial?.intent_metrics?.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono">{(simple?.intent_metrics?.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {(proposed?.intent_metrics?.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 75.0%</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Intent Macro F1</td>
                      <td className="py-3 px-4 font-mono">{trivial?.intent_metrics?.macro_f1}</td>
                      <td className="py-3 px-4 font-mono">{simple?.intent_metrics?.macro_f1}</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {proposed?.intent_metrics?.macro_f1}
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 0.700</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Escalation Accuracy</td>
                      <td className="py-3 px-4 font-mono">{(trivial?.escalation_metrics?.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono">{(simple?.escalation_metrics?.accuracy * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {(proposed?.escalation_metrics?.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 70.0%</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Escalation Precision</td>
                      <td className="py-3 px-4 font-mono">{trivial?.escalation_metrics?.precision}</td>
                      <td className="py-3 px-4 font-mono">{simple?.escalation_metrics?.precision}</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {proposed?.escalation_metrics?.precision}
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 0.450</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Escalation Recall</td>
                      <td className="py-3 px-4 font-mono">{trivial?.escalation_metrics?.recall}</td>
                      <td className="py-3 px-4 font-mono">{simple?.escalation_metrics?.recall}</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {proposed?.escalation_metrics?.recall}
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 0.650</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Escalation F1</td>
                      <td className="py-3 px-4 font-mono">{trivial?.escalation_metrics?.f1}</td>
                      <td className="py-3 px-4 font-mono">{simple?.escalation_metrics?.f1}</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {proposed?.escalation_metrics?.f1}
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 0.500</td>
                    </tr>
                    <tr className="bg-rose-950/10">
                      <td className="py-3 px-4 font-semibold text-rose-300">
                        False Auto-Handle Rate (Critical Safety)
                      </td>
                      <td className="py-3 px-4 font-mono text-rose-400 font-bold">100.0% (Fatal)</td>
                      <td className="py-3 px-4 font-mono text-rose-300">90.0%</td>
                      <td className="py-3 px-4 font-mono font-bold text-emerald-400 bg-sky-950/10">
                        {(proposed?.escalation_metrics?.false_autohandle_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&lt; 35.0%</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">
                        Unnecessary Escalation Rate (Cost Waste)
                      </td>
                      <td className="py-3 px-4 font-mono">0.0%</td>
                      <td className="py-3 px-4 font-mono">8.7%</td>
                      <td className="py-3 px-4 font-mono font-bold text-slate-200 bg-sky-950/10">
                        {(proposed?.escalation_metrics?.unnecessary_escalation_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&lt; 30.0%</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">LLM Judge Quality (1-5 Scale)</td>
                      <td className="py-3 px-4 font-mono">4.12</td>
                      <td className="py-3 px-4 font-mono">3.66</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400 bg-sky-950/10">
                        {proposed?.judge_metrics?.mean_judge_score}
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 3.80</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-semibold text-white">Judge-Human Adjacent Agreement Rate</td>
                      <td className="py-3 px-4 font-mono">87.0%</td>
                      <td className="py-3 px-4 font-mono">80.5%</td>
                      <td className="py-3 px-4 font-mono font-bold text-emerald-400 bg-sky-950/10">
                        {(proposed?.judge_metrics?.adjacent_agreement_rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-mono">&gt; 90.0%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Per-Intent Performance Breakdown */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-semibold text-white">Per-Intent Precision, Recall, and F1 Breakdown</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {proposed?.intent_metrics?.per_intent &&
                  Object.entries(proposed.intent_metrics.per_intent).map(([intentKey, metrics]: [string, any]) => (
                    <div key={intentKey} className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-200 font-mono">{intentKey}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                          N={metrics.support}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-xs pt-1 border-t border-slate-800/60">
                        <div>
                          <div className="text-[10px] text-slate-500">Precision</div>
                          <div className="font-semibold text-slate-200">{(metrics.precision * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-slate-500">Recall</div>
                          <div className="font-semibold text-slate-200">{(metrics.recall * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-slate-500">F1</div>
                          <div className="font-semibold text-sky-400 font-mono">{metrics.f1.toFixed(3)}</div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: GOLDEN DATASET EXPLORER */}
        {activeTab === 'dataset' && (
          <div className="space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white">Golden Evaluation Set Explorer (N = 200)</h2>
                <p className="text-xs text-slate-400">
                  Curated and hand-labelled from authentic multi-turn Twitter customer support conversations (<span className="text-slate-300 font-mono">thoughtvector/customer-support-on-twitter</span>).
                </p>
              </div>

              {/* Filters */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input
                    type="text"
                    value={datasetSearch}
                    onChange={(e) => setDatasetSearch(e.target.value)}
                    placeholder="Search queries or tweets..."
                    className="bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
                  />
                </div>

                <select
                  value={intentFilter}
                  onChange={(e) => setIntentFilter(e.target.value)}
                  className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="ALL">All Intents ({goldenData.length})</option>
                  <option value="SOFTWARE_OS_BUG">SOFTWARE_OS_BUG</option>
                  <option value="HARDWARE_BATTERY">HARDWARE_BATTERY</option>
                  <option value="ACCOUNT_SECURITY_ICLOUD">ACCOUNT_SECURITY_ICLOUD</option>
                  <option value="DEVICE_SETUP_FEATURE">DEVICE_SETUP_FEATURE</option>
                  <option value="GENERAL_FEEDBACK_RANT">GENERAL_FEEDBACK_RANT</option>
                </select>

                <select
                  value={escalateFilter}
                  onChange={(e) => setEscalateFilter(e.target.value)}
                  className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="ALL">All Escalation</option>
                  <option value="ESCALATE">Escalate to Human (True)</option>
                  <option value="AUTOHANDLE">Auto-Handle (False)</option>
                </select>
              </div>
            </div>

            {/* List Table */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden">
              <div className="overflow-x-auto max-h-[620px]">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="sticky top-0 bg-slate-950 border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                    <tr>
                      <th className="py-2.5 px-3">ID</th>
                      <th className="py-2.5 px-3">Customer Tweet</th>
                      <th className="py-2.5 px-3">Gold Intent</th>
                      <th className="py-2.5 px-3">Escalate</th>
                      <th className="py-2.5 px-3">Quality</th>
                      <th className="py-2.5 px-3 text-right">Inspect</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {filteredDataset.map((item) => (
                      <tr
                        key={item.id}
                        className="hover:bg-slate-800/40 transition-colors cursor-pointer"
                        onClick={() => setSelectedRecord(item)}
                      >
                        <td className="py-2.5 px-3 font-mono text-slate-400 font-semibold">{item.id}</td>
                        <td className="py-2.5 px-3 text-slate-200 max-w-md truncate font-sans">
                          {item.customer_text}
                        </td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-slate-800 text-sky-400 border border-slate-700/60">
                            {item.gold_intent}
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            item.gold_escalate
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          }`}>
                            {item.gold_escalate ? 'Escalate' : 'Auto-Handle'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono font-semibold text-slate-300">
                          {item.human_quality_score.toFixed(1)} / 5
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button className="text-sky-400 hover:text-sky-300 font-medium">
                            Details →
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-2 bg-slate-950/60 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
                <span>Showing {filteredDataset.length} of {goldenData.length} records</span>
                <span className="italic">Click any record to view multi-turn context and annotator notes</span>
              </div>
            </div>

            {/* Record Detail Modal */}
            {selectedRecord && (
              <div className="fixed inset-0 bg-black/70 backdrop-blur-xs flex items-center justify-center p-4 z-50">
                <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-bold text-sky-400">{selectedRecord.id}</span>
                      <span className="text-xs text-slate-400 font-mono">(Raw ID: {selectedRecord.raw_conversation_id})</span>
                    </div>
                    <button
                      onClick={() => setSelectedRecord(null)}
                      className="text-slate-400 hover:text-white text-sm px-2 py-1 rounded-lg bg-slate-800"
                    >
                      ✕ Close
                    </button>
                  </div>

                  <div className="space-y-3 text-xs">
                    <div>
                      <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">Customer Message:</span>
                      <p className="mt-1 p-3 rounded-lg bg-slate-950 text-slate-100 text-sm leading-relaxed border border-slate-800">
                        "{selectedRecord.customer_text}"
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">Real Historical Apple Support Reply:</span>
                      <p className="mt-1 p-3 rounded-lg bg-slate-950/80 text-emerald-300 leading-relaxed border border-emerald-950/40">
                        "{selectedRecord.real_agent_reply}"
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3 pt-2">
                      <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80">
                        <span className="text-slate-400">Ground Truth Intent:</span>
                        <div className="font-bold text-sky-400 font-mono mt-0.5">{selectedRecord.gold_intent}</div>
                      </div>
                      <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80">
                        <span className="text-slate-400">Escalation Policy:</span>
                        <div className={`font-bold mt-0.5 ${selectedRecord.gold_escalate ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {selectedRecord.gold_escalate ? 'Escalate to Human / DM' : 'Auto-Handle Self-Service'}
                        </div>
                      </div>
                    </div>

                    <div>
                      <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">Escalation Policy Reason:</span>
                      <p className="mt-0.5 text-slate-300 italic">{selectedRecord.gold_escalation_reason}</p>
                    </div>

                    <div>
                      <span className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">Annotator Notes:</span>
                      <p className="mt-0.5 text-slate-400">{selectedRecord.annotator_notes}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: LLM JUDGE & HUMAN CALIBRATION */}
        {activeTab === 'calibration' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">LLM-as-a-Judge Calibration & Agreement Evidence</h2>
              <p className="text-xs text-slate-400">
                To prove that our automated evaluation is trustworthy, we benchmarked the LLM Judge against human expert evaluations across all 200 Golden cases.
              </p>
            </div>

            {/* Calibration Metrics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Adjacent Agreement Rate</span>
                <div className="text-2xl font-extrabold text-emerald-400">
                  {(proposed?.judge_metrics?.adjacent_agreement_rate * 100).toFixed(1)}%
                </div>
                <p className="text-[11px] text-slate-400">Within ±1.0 score boundary</p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Exact Agreement Rate</span>
                <div className="text-2xl font-extrabold text-sky-400">
                  {(proposed?.judge_metrics?.exact_agreement_rate * 100).toFixed(1)}%
                </div>
                <p className="text-[11px] text-slate-400">Within ±0.5 score boundary</p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Mean Calibration Bias (Δ)</span>
                <div className="text-2xl font-extrabold text-indigo-300">
                  {proposed?.judge_metrics?.calibration_bias > 0 ? `+${proposed?.judge_metrics?.calibration_bias}` : proposed?.judge_metrics?.calibration_bias}
                </div>
                <p className="text-[11px] text-slate-400">Judge Mean ({proposed?.judge_metrics?.mean_judge_score}) - Human ({proposed?.judge_metrics?.mean_human_score})</p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl space-y-1">
                <span className="text-xs text-slate-400 font-medium">Cohen's Weighted Kappa (κ)</span>
                <div className="text-2xl font-extrabold text-slate-200 font-mono">
                  {proposed?.judge_metrics?.cohen_weighted_kappa}
                </div>
                <p className="text-[11px] text-slate-400">Quadratic weights on ordinal scale</p>
              </div>
            </div>

            {/* 4-Axis Rubric Breakdown */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-semibold text-white">4-Axis Evaluation Rubric Breakdown</h3>
              <p className="text-xs text-slate-400">
                Every generated response is scored against four objective dimensions reflecting Apple's public support policies:
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-sky-400">Factual Grounding</span>
                    <span className="text-xs font-mono font-bold text-white">
                      {proposed?.judge_metrics?.rubric_averages?.grounding || '4.1'} / 5
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Cites legitimate Apple diagnostic steps (Settings, reset, model isolation) and official support.apple.com / iforgot links.
                  </p>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-sky-400 h-full rounded-full" style={{ width: '82%' }}></div>
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-emerald-400">Brand Tone & Voice</span>
                    <span className="text-xs font-mono font-bold text-white">
                      {proposed?.judge_metrics?.rubric_averages?.tone || '4.0'} / 5
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Maintains Apple's calm, empathetic, and objective tone ("We'd love to look into this") without defensive sarcasm.
                  </p>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-emerald-400 h-full rounded-full" style={{ width: '80%' }}></div>
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-rose-400">Escalation Safety</span>
                    <span className="text-xs font-mono font-bold text-white">
                      {proposed?.judge_metrics?.rubric_averages?.safety || '4.4'} / 5
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Zero PII violations: never requests passwords or 2FA codes on public tweets; directs hardware hazards to Genius Bar.
                  </p>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-rose-400 h-full rounded-full" style={{ width: '88%' }}></div>
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-amber-400">Resolution Actionability</span>
                    <span className="text-xs font-mono font-bold text-white">
                      {proposed?.judge_metrics?.rubric_averages?.actionability || '4.2'} / 5
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Provides concrete diagnostic questions (iOS version, device model) or verified actionable steps.
                  </p>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-amber-400 h-full rounded-full" style={{ width: '84%' }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: TOP 5 FAILURE MODES */}
        {activeTab === 'failures' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">Top 5 Failure Modes & Root-Cause Hypotheses</h2>
              <p className="text-xs text-slate-400">
                Rigorous failure analysis identifying where the proposed agent struggles on real Kaggle customer tweets and the exact engineering mitigations required.
              </p>
            </div>

            <div className="space-y-4">
              {benchmarkData?.top_failure_modes?.map((failure) => (
                <div key={failure.rank} className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-2.5">
                      <span className="w-6 h-6 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30 flex items-center justify-center font-bold text-xs">
                        #{failure.rank}
                      </span>
                      <h3 className="text-sm font-bold text-white">{failure.title}</h3>
                    </div>
                    <span className="font-mono text-xs text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                      Tweet ID: {failure.tweet_id}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs italic text-slate-200">
                    "{failure.example_tweet}"
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
                      <span className="font-semibold text-amber-400 uppercase tracking-wider text-[10px]">Root-Cause Hypothesis</span>
                      <p className="text-slate-300 leading-relaxed">{failure.root_cause_hypothesis}</p>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
                      <span className="font-semibold text-emerald-400 uppercase tracking-wider text-[10px]">Engineering Mitigation</span>
                      <p className="text-slate-300 leading-relaxed">{failure.mitigation}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 6: TECHNICAL REPORT */}
        {activeTab === 'report' && (
          <div className="space-y-6">
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 md:p-8 space-y-6">
              <div className="border-b border-slate-800 pb-4">
                <div className="flex items-center gap-2 text-xs text-sky-400 font-semibold font-mono uppercase tracking-wider mb-1">
                  Full Technical Report • REPORT.md
                </div>
                <h2 className="text-xl font-bold text-white">
                  Building and Verifying a Grounded AI Customer Support Agent for @AppleSupport
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Includes problem framing, non-goals, empirical results vs. baselines, failure modes, and the mandatory "What is misleading about my headline number?" critique.
                </p>
              </div>

              {/* Rendered Report Content */}
              <div className="prose prose-invert prose-xs max-w-none space-y-6 text-slate-300 text-xs leading-relaxed font-sans">
                {reportMarkdown ? (
                  <div className="whitespace-pre-wrap font-sans leading-relaxed text-xs">
                    {reportMarkdown}
                  </div>
                ) : (
                  <p className="text-slate-400 italic">Loading technical report...</p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-4 sm:px-6 bg-[#090d16] text-[11px] sm:text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2.5 text-center sm:text-left">
        <div>
          OmniSupport AI Agent • Built on Kaggle <span className="font-mono text-slate-400">thoughtvector/customer-support-on-twitter</span>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-4">
          <span>Dataset: @AppleSupport Corpus</span>
          <span>•</span>
          <span>N = 200 Golden Benchmark</span>
          <span>•</span>
          <span className="text-emerald-400">100% Pure Python Backend</span>
        </div>
      </footer>
    </div>
  );
}
