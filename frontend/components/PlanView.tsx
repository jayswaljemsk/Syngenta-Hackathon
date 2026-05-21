'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import type { CSSProperties } from 'react';
import type { Anomaly, Language, Outlet, PlanData, PlanMode } from '@/lib/types';
import { useT } from '@/lib/i18n';
import BottomSheet from './BottomSheet';

const MapView = dynamic(() => import('./MapView'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[420px] w-full items-center justify-center bg-slate-100 sm:min-h-0">
      <span className="text-sm font-medium text-slate-400">Loading map...</span>
    </div>
  ),
});

const LANG_OPTIONS: { code: Language; label: string }[] = [
  { code: 'en', label: 'EN' },
  { code: 'hi', label: 'HI' },
  { code: 'mr', label: 'MR' },
];

const ANOMALY_SEVERITY_STYLES: Record<Anomaly['severity'], string> = {
  high: 'border-red-200 bg-red-50 text-red-800',
  medium: 'border-orange-200 bg-orange-50 text-orange-800',
  low: 'border-yellow-200 bg-yellow-50 text-yellow-800',
  info: 'border-blue-200 bg-blue-50 text-blue-800',
};

const loginStyles: Record<string, CSSProperties> = {
  page: {
    minHeight: '100svh',
    background: '#eef3ef',
    color: '#020618',
    padding: 'clamp(12px, 3vw, 24px)',
    display: 'flex',
    alignItems: 'center',
  },
  card: {
    width: 'min(100%, 1040px)',
    minHeight: 'min(720px, calc(100svh - 24px))',
    margin: '0 auto',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 390px), 1fr))',
    overflow: 'hidden',
    borderRadius: 24,
    background: '#ffffff',
    boxShadow: '0 28px 80px rgba(0, 44, 34, 0.14)',
  },
  hero: {
    background: '#0a6e3d',
    color: '#ffffff',
    padding: 'clamp(28px, 5vw, 48px)',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  kicker: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.22em',
    fontSize: 12,
    fontWeight: 700,
    color: 'rgba(255, 255, 255, 0.68)',
  },
  title: {
    margin: '32px 0 0',
    maxWidth: 520,
    fontSize: 'clamp(36px, 4vw, 56px)',
    lineHeight: 1.02,
    fontWeight: 800,
  },
  copy: {
    margin: '20px 0 0',
    maxWidth: 460,
    color: 'rgba(255, 255, 255, 0.78)',
    fontSize: 16,
    lineHeight: 1.65,
  },
  stats: {
    marginTop: 48,
    display: 'grid',
    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
    gap: 12,
  },
  stat: {
    borderRadius: 18,
    background: 'rgba(255, 255, 255, 0.13)',
    padding: 14,
    textAlign: 'center',
  },
  formWrap: {
    display: 'flex',
    alignItems: 'center',
    padding: 'clamp(28px, 5vw, 48px)',
  },
  form: {
    width: '100%',
  },
  formKicker: {
    margin: 0,
    textTransform: 'uppercase',
    letterSpacing: '0.18em',
    fontSize: 12,
    fontWeight: 800,
    color: '#007956',
  },
  label: {
    display: 'block',
    marginTop: 28,
    color: '#314158',
    fontSize: 14,
    fontWeight: 700,
  },
  input: {
    width: '100%',
    marginTop: 8,
    border: '1px solid #e2e8f0',
    borderRadius: 18,
    background: '#f8fafc',
    padding: '14px 16px',
    color: '#020618',
    fontSize: 18,
    fontWeight: 700,
    outline: 'none',
  },
  territory: {
    marginTop: 18,
    border: '1px solid #e2e8f0',
    borderRadius: 18,
    background: '#f8fafc',
    padding: 16,
  },
  button: {
    width: '100%',
    marginTop: 24,
    border: 0,
    borderRadius: 18,
    background: '#0a6e3d',
    color: '#ffffff',
    padding: '15px 20px',
    fontSize: 15,
    fontWeight: 800,
    cursor: 'pointer',
    boxShadow: '0 16px 36px rgba(0, 78, 59, 0.18)',
  },
};

function formatSyncTime(syncedAt: string): string {
  try {
    return new Date(syncedAt).toLocaleTimeString('en-IN', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return '5:12 AM';
  }
}

function pinClass(pos: number): string {
  if (pos <= 3) return 'bg-red-600 text-white';
  if (pos <= 7) return 'bg-amber-600 text-white';
  return 'bg-green-600 text-white';
}

function TopBar({
  plan,
  lang,
  onLangChange,
  t,
}: {
  plan: PlanData;
  lang: Language;
  onLangChange: (l: Language) => void;
  t: ReturnType<typeof useT>;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-t-[1.35rem] bg-[#0A6E3D] px-4 py-2.5 text-white sm:rounded-t-3xl sm:px-5">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-white/70">{plan.rep_id}</p>
        <p className="text-base font-semibold leading-tight">{plan.territory}</p>
      </div>

      <div className="flex items-center gap-2 rounded-full bg-white/15 px-3 py-1.5 text-xs font-medium">
        <span className="h-2 w-2 rounded-full bg-emerald-300 shadow-[0_0_0_3px_rgba(110,231,183,0.18)]" />
        <span>{t('synced')} {formatSyncTime(plan.synced_at)} - {t('offline')}</span>
      </div>

      <div className="flex items-center gap-1 rounded-full bg-white/15 p-1">
        {LANG_OPTIONS.map((opt) => (
          <button
            key={opt.code}
            type="button"
            onClick={() => onLangChange(opt.code)}
            className={`min-w-9 rounded-full px-2.5 py-1 text-xs font-semibold transition ${
              lang === opt.code ? 'bg-white text-[#0A6E3D]' : 'text-white hover:bg-white/15'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function DemoStart({
  plan,
  onStart,
}: {
  plan: PlanData;
  onStart: () => void;
}) {
  return (
    <main style={loginStyles.page}>
      <section style={loginStyles.card}>
        <div style={loginStyles.hero}>
          <div>
            <p style={loginStyles.kicker}>Syngenta field intelligence</p>
            <h1 style={loginStyles.title}>Daily route plan for every rep.</h1>
            <p style={loginStyles.copy}>
              Ranked retailer visits, route order, product recommendation, and offline-ready explainability across Rabi 2025-26.
            </p>
          </div>

          <div style={loginStyles.stats}>
            <div style={loginStyles.stat}>
              <p style={{ margin: 0, fontSize: 28, lineHeight: 1, fontWeight: 800 }}>10</p>
              <span style={{ display: 'block', marginTop: 8, color: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }}>outlets</span>
            </div>
            <div style={loginStyles.stat}>
              <p style={{ margin: 0, fontSize: 28, lineHeight: 1, fontWeight: 800 }}>5:12</p>
              <span style={{ display: 'block', marginTop: 8, color: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }}>sync</span>
            </div>
            <div style={loginStyles.stat}>
              <p style={{ margin: 0, fontSize: 28, lineHeight: 1, fontWeight: 800 }}>+2d</p>
              <span style={{ display: 'block', marginTop: 8, color: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }}>rain sim</span>
            </div>
          </div>
        </div>

        <div style={loginStyles.formWrap}>
          <div style={loginStyles.form}>
            <p style={loginStyles.formKicker}>Rep login</p>
            <label style={loginStyles.label} htmlFor="rep-id">Rep ID</label>
            <input
              id="rep-id"
              value={plan.rep_id}
              readOnly
              suppressHydrationWarning
              style={loginStyles.input}
            />
            <div style={loginStyles.territory}>
              <p style={{ margin: 0, color: '#0f172b', fontSize: 15, fontWeight: 800 }}>{plan.territory}</p>
              <span style={{ display: 'block', marginTop: 8, color: '#45556c', fontSize: 14, lineHeight: 1.6 }}>{plan.weather_summary}</span>
            </div>
            <button
              type="button"
              onClick={onStart}
              style={loginStyles.button}
            >
              Today&apos;s Plan
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

function DateToggle({
  mode,
  onChange,
  activeDate,
  t,
}: {
  mode: PlanMode;
  onChange: (m: PlanMode) => void;
  activeDate: string;
  t: ReturnType<typeof useT>;
}) {
  const dateLabel = useMemo(() => {
    const parsed = new Date(`${activeDate}T00:00:00`);
    if (Number.isNaN(parsed.getTime())) return activeDate;
    return parsed.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
  }, [activeDate]);

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 sm:px-5">
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{t('view')}</span>
      {(['today', 'post_rain'] as PlanMode[]).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={`rounded-full border px-4 py-2 text-xs font-bold transition ${
            mode === m
              ? 'border-[#0A6E3D] bg-[#0A6E3D] text-white shadow-sm'
              : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-[#0A6E3D] hover:text-[#0A6E3D]'
          }`}
        >
          {m === 'today' ? `${t('todayLabel')} - ${dateLabel}` : t('postRainLabel')}
        </button>
      ))}
    </div>
  );
}

function OutcomeForm({
  outlet,
  repId,
  date,
  onClose,
  onSubmit,
  t,
}: {
  outlet: Outlet;
  repId: string;
  date: string;
  onClose: () => void;
  onSubmit: (outletId: string, payload: Record<string, unknown>) => Promise<void>;
  t: ReturnType<typeof useT>;
}) {
  const [saleMade, setSaleMade] = useState(true);
  const [saleValue, setSaleValue] = useState(String(outlet.recommended_product?.expected_uplift_inr ?? ''));
  const [dismissalReason, setDismissalReason] = useState('');
  const [notes, setNotes] = useState('');

  const product = outlet.recommended_product;

  return (
    <div className="fixed inset-0 z-[2000] flex items-end justify-center bg-slate-950/35 p-3 sm:items-center">
      <form
        className="w-full max-w-md rounded-2xl bg-white p-4 shadow-2xl sm:p-5"
        onSubmit={async (event) => {
          event.preventDefault();
          const payload = {
            rep_id: repId,
            outlet_id: outlet.outlet_id,
            date,
            sale_made: saleMade,
            sale_value_inr: saleMade ? Number(saleValue || 0) : 0,
            product_id: product?.product_id ?? null,
            dismissal_reason: saleMade ? null : dismissalReason || 'No immediate need',
            rep_notes: notes || null,
          };
          await onSubmit(outlet.outlet_id, payload);
        }}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-emerald-700">{t('outcomeTitle')}</p>
            <h2 className="mt-1 text-lg font-bold text-slate-950">{outlet.name}</h2>
            <p className="mt-0.5 text-sm text-slate-500">{product?.product_name ?? t('noProduct')}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-100 text-sm font-bold text-slate-500"
            aria-label="Close outcome form"
          >
            X
          </button>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setSaleMade(true)}
            className={`rounded-xl border px-3 py-2 text-sm font-bold ${saleMade ? 'border-[#0A6E3D] bg-[#0A6E3D] text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
          >
            {t('saleMade')}
          </button>
          <button
            type="button"
            onClick={() => setSaleMade(false)}
            className={`rounded-xl border px-3 py-2 text-sm font-bold ${!saleMade ? 'border-[#0A6E3D] bg-[#0A6E3D] text-white' : 'border-slate-200 bg-slate-50 text-slate-700'}`}
          >
            {t('noSale')}
          </button>
        </div>

        {saleMade ? (
          <label className="mt-4 block text-sm font-semibold text-slate-700">
            {t('saleValue')}
            <input
              type="number"
              min="0"
              value={saleValue}
              onChange={(event) => setSaleValue(event.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-950 outline-none"
            />
          </label>
        ) : (
          <label className="mt-4 block text-sm font-semibold text-slate-700">
            {t('dismissalReason')}
            <select
              value={dismissalReason}
              onChange={(event) => setDismissalReason(event.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-950 outline-none"
            >
              <option value="">{t('selectReason')}</option>
              <option value="stock_available">{t('reasonStock')}</option>
              <option value="price_sensitive">{t('reasonPrice')}</option>
              <option value="follow_up_later">{t('reasonFollowUp')}</option>
            </select>
          </label>
        )}

        <label className="mt-4 block text-sm font-semibold text-slate-700">
          {t('repNotes')}
          <textarea
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            rows={3}
            className="mt-1 w-full resize-none rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-950 outline-none"
            placeholder={t('notesPlaceholder')}
          />
        </label>

        <button
          type="submit"
          className="mt-4 w-full rounded-xl bg-[#0A6E3D] px-4 py-3 text-sm font-bold text-white"
        >
          {t('saveOutcome')}
        </button>
      </form>
    </div>
  );
}

function OutcomeModal({
  outlet,
  repId,
  date,
  onClose,
  onSubmit,
  t,
}: {
  outlet: Outlet | null;
  repId: string;
  date: string;
  onClose: () => void;
  onSubmit: (outletId: string, payload: Record<string, unknown>) => Promise<void>;
  t: ReturnType<typeof useT>;
}) {
  if (!outlet) return null;

  return (
    <OutcomeForm
      key={outlet.outlet_id}
      outlet={outlet}
      repId={repId}
      date={date}
      onClose={onClose}
      onSubmit={onSubmit}
      t={t}
    />
  );
}

function WeatherBanner({ summary }: { summary: string }) {
  return (
    <div className="overflow-x-auto border-b border-amber-200 bg-amber-50 px-4 py-1.5 sm:px-5">
      <p className="w-max whitespace-nowrap text-sm font-medium text-amber-900">{summary}</p>
    </div>
  );
}

function translateAnomaly(type: string, fallback: string, t: ReturnType<typeof useT>): string {
  if (type === 'early_pest_emergence') return t('anomalyWhitefly');
  if (type === 'demand_spike') return t('anomalyFungicide');
  return fallback;
}

function AnomalyBar({ anomalies, t }: { anomalies: Anomaly[]; t: ReturnType<typeof useT> }) {
  if (anomalies.length === 0) return null;

  return (
    <div className="flex gap-2 overflow-x-auto border-b border-slate-100 bg-slate-50 px-4 py-1.5 sm:px-5">
      {anomalies.map((a) => (
        <div
          key={`${a.type}-${a.description}`}
          className={`shrink-0 rounded-full border px-3 py-1 text-xs font-semibold ${ANOMALY_SEVERITY_STYLES[a.severity]}`}
        >
          {translateAnomaly(a.type, a.description, t)}
        </div>
      ))}
    </div>
  );
}

function MapLegend({ t }: { t: ReturnType<typeof useT> }) {
  const items = [
    { color: 'bg-red-600', label: t('topPriority') },
    { color: 'bg-amber-600', label: t('midPriority') },
    { color: 'bg-green-600', label: t('lowPriority') },
  ];

  return (
    <div className="absolute right-3 top-3 z-[500] rounded-2xl border border-slate-200 bg-white/95 px-3 py-2 shadow-lg shadow-slate-900/10 backdrop-blur">
      {items.map((item) => (
        <div key={item.label} className="mb-1 flex items-center gap-2 text-xs font-medium text-slate-700 last:mb-0">
          <span className={`h-2.5 w-2.5 rounded-full ${item.color}`} />
          {item.label}
        </div>
      ))}
    </div>
  );
}

function RoutePanel({
  outlets,
  selectedOutletId,
  visitedIds,
  onSelect,
  t,
}: {
  outlets: Outlet[];
  selectedOutletId: string | null;
  visitedIds: Set<string>;
  onSelect: (outlet: Outlet) => void;
  t: ReturnType<typeof useT>;
}) {
  return (
    <aside className="hidden border-l border-slate-200 bg-white lg:flex lg:min-h-0 lg:w-[360px] lg:flex-col">
      <div className="border-b border-slate-100 p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">{t('todaysRoute')}</p>
        <p className="mt-1 text-sm text-slate-500">{t('routePanelHint')}</p>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {outlets
          .slice()
          .sort((a, b) => a.route_position - b.route_position)
          .map((outlet) => {
            const selected = outlet.outlet_id === selectedOutletId;
            const visited = visitedIds.has(outlet.outlet_id);

            return (
              <button
                key={outlet.outlet_id}
                type="button"
                onClick={() => onSelect(outlet)}
                className={`mb-2 w-full rounded-2xl border p-3 text-left transition ${
                  selected
                    ? 'border-emerald-300 bg-emerald-50 shadow-sm'
                    : 'border-slate-200 bg-white hover:border-emerald-200 hover:bg-slate-50'
                } ${visited ? 'opacity-70' : ''}`}
              >
                <div className="flex items-start gap-3">
                  <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${visited ? 'bg-slate-300 text-slate-700' : pinClass(outlet.route_position)}`}>
                    {visited ? 'OK' : outlet.route_position}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-bold text-slate-900">{outlet.name}</p>
                    <p className="mt-0.5 truncate text-xs text-slate-500">{outlet.address}</p>
                    <div className="mt-2 flex items-center justify-between gap-2">
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                        {t('score')} {outlet.priority_score.toFixed(2)}
                      </span>
                      <span className="text-[11px] font-medium text-slate-500">{outlet.estimated_visit_minutes} min</span>
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
      </div>
    </aside>
  );
}

interface PlanViewProps {
  todayPlan: PlanData;
  rainPlan: PlanData;
}

export default function PlanView({ todayPlan, rainPlan }: PlanViewProps) {
  const [started, setStarted] = useState(false);
  const [mode, setMode] = useState<PlanMode>('today');
  const [lang, setLang] = useState<Language>('en');
  const [selectedOutlet, setSelected] = useState<Outlet | null>(null);
  const [outcomeOutlet, setOutcomeOutlet] = useState<Outlet | null>(null);
  const [visitedIds, setVisitedIds] = useState<Set<string>>(new Set());
  const [reshuffleNotice, setReshuffleNotice] = useState(false);

  const t = useT(lang);
  const plan = mode === 'today' ? todayPlan : rainPlan;
  const translatedWeather = mode === 'today' ? t('weatherToday') : t('weatherPostRain');
  const topOutlet = useMemo(
    () => plan.outlets.reduce((best, outlet) => (outlet.route_position < best.route_position ? outlet : best), plan.outlets[0]),
    [plan.outlets],
  );

  useEffect(() => {
    if (!reshuffleNotice) return;
    const timer = window.setTimeout(() => setReshuffleNotice(false), 2600);
    return () => window.clearTimeout(timer);
  }, [reshuffleNotice]);

  const handleModeChange = useCallback((m: PlanMode) => {
    setSelected(null);
    setMode(m);
    setReshuffleNotice(m === 'post_rain');
  }, []);

  const handleOpenOutcome = useCallback((outlet: Outlet) => {
    setOutcomeOutlet(outlet);
  }, []);

  const handleSubmitOutcome = useCallback(async (id: string, payload: Record<string, unknown>) => {
    try {
      await fetch('/api/outcome', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } finally {
      setVisitedIds((prev) => new Set([...prev, id]));
      setOutcomeOutlet(null);
    }
  }, []);

  if (!started) {
    return <DemoStart plan={todayPlan} onStart={() => setStarted(true)} />;
  }

  return (
    <main className="h-[100svh] overflow-hidden bg-[#eef3ef] p-2 text-slate-950 sm:p-3 lg:p-4">
      <section className="mx-auto flex h-full min-h-0 max-w-7xl flex-col overflow-hidden rounded-[1.35rem] bg-white shadow-2xl shadow-emerald-950/10 sm:rounded-3xl">
        <TopBar plan={plan} lang={lang} onLangChange={setLang} t={t} />
        <DateToggle mode={mode} onChange={handleModeChange} activeDate={plan.date} t={t} />
        <WeatherBanner summary={translatedWeather} />
        <AnomalyBar anomalies={plan.anomalies} t={t} />

        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="relative min-h-0 flex-1 overflow-hidden bg-slate-100">
            <MapView
              plan={plan}
              onOutletSelect={setSelected}
              visitedIds={visitedIds}
              selectedOutletId={selectedOutlet?.outlet_id ?? null}
            />
            <MapLegend t={t} />

            <div className="absolute left-16 top-3 z-[500] max-w-[calc(100%-11rem)] rounded-2xl border border-white/70 bg-white/95 px-3 py-2 shadow-lg shadow-slate-900/10 backdrop-blur">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-700">{t('nextStop')}</p>
              <p className="mt-0.5 truncate text-sm font-bold text-slate-900">#{topOutlet.route_position} {topOutlet.name}</p>
            </div>

            <div
              className={`absolute left-1/2 top-20 z-[700] w-[min(92%,380px)] -translate-x-1/2 rounded-2xl border border-emerald-200 bg-white px-4 py-3 text-sm font-semibold text-emerald-900 shadow-xl shadow-emerald-950/15 transition-all duration-300 ${
                reshuffleNotice ? 'translate-y-0 opacity-100' : '-translate-y-3 opacity-0 pointer-events-none'
              }`}
            >
              {t('reshuffleNotice')} {t('newNumberOne')} {topOutlet.name}.
            </div>

            <BottomSheet
              outlet={selectedOutlet}
              onClose={() => setSelected(null)}
              onMarkVisited={handleOpenOutcome}
              t={t}
            />
          </div>

          <RoutePanel
            outlets={plan.outlets}
            selectedOutletId={selectedOutlet?.outlet_id ?? null}
            visitedIds={visitedIds}
            onSelect={setSelected}
            t={t}
          />
        </div>

        <div className="flex shrink-0 items-center justify-between gap-3 border-t border-slate-200 bg-white px-4 py-2 text-xs text-slate-500 sm:px-5">
          <span>{visitedIds.size} / {plan.outlets.length} {t('outletsVisited')}</span>
          <span className="hidden sm:inline">{t('footerNote')}</span>
        </div>

        <OutcomeModal
          outlet={outcomeOutlet}
          repId={plan.rep_id}
          date={plan.date}
          onClose={() => setOutcomeOutlet(null)}
          onSubmit={handleSubmitOutcome}
          t={t}
        />
      </section>
    </main>
  );
}
