'use client';

import type { Outlet } from '@/lib/types';
import type { TFunction } from '@/lib/i18n';

interface BottomSheetProps {
  outlet: Outlet | null;
  onClose: () => void;
  onMarkVisited: (outlet: Outlet) => void;
  t: TFunction;
}

const CONFIDENCE_STYLES: Record<string, string> = {
  High: 'bg-green-100 text-green-800',
  Medium: 'bg-yellow-100 text-yellow-800',
  Low: 'bg-red-100 text-red-800',
};

const CATEGORY_INITIALS: Record<string, string> = {
  Fungicide: 'F',
  Insecticide: 'I',
  Herbicide: 'H',
};

export default function BottomSheet({
  outlet,
  onClose,
  onMarkVisited,
  t,
}: BottomSheetProps) {
  if (!outlet) return null;

  return (
    <>
      <div
        className="absolute inset-0 z-[900] bg-slate-950/20 transition-opacity duration-300"
        onClick={onClose}
        aria-hidden="true"
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-label={outlet.name}
        className="absolute bottom-0 left-0 right-0 z-[1000] flex max-h-[82%] flex-col rounded-t-[1.35rem] bg-white shadow-2xl transition-transform duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] lg:bottom-4 lg:left-4 lg:right-4 lg:max-h-[72%] lg:rounded-2xl"
      >
        <div className="flex shrink-0 justify-center pb-1 pt-3">
          <div className="h-1 w-10 rounded-full bg-slate-300" />
        </div>

        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-3 flex h-8 w-8 items-center justify-center rounded-full bg-slate-100 text-lg leading-none text-slate-500 transition hover:bg-slate-200"
          aria-label="Close panel"
        >
          X
        </button>

        <div className="shrink-0 border-b border-slate-100 px-4 pb-3 sm:px-5">
          <p className="pr-10 text-lg font-bold leading-snug text-slate-950">{outlet.name}</p>
          <p className="mt-0.5 text-sm text-slate-500">{outlet.address}</p>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${CONFIDENCE_STYLES[outlet.confidence]}`}>
              {t(outlet.confidence.toLowerCase())} {t('confidence')}
            </span>
            <span className="text-xs font-medium text-slate-500">
              {t('routeStop')} #{outlet.route_position} - {t('score')} {outlet.priority_score.toFixed(2)} - {outlet.estimated_visit_minutes} {t('visitMins')}
            </span>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-3 pt-4 sm:px-5">
          <ul className="mb-4 space-y-2.5">
            {outlet.reason_strings.map((reason) => (
              <li key={reason} className="flex items-start gap-2.5">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[#0A6E3D]" aria-hidden="true" />
                <span className="text-sm leading-6 text-slate-700">{reason}</span>
              </li>
            ))}
          </ul>

          {outlet.recommended_product ? (
            <div className="mb-3 rounded-2xl border border-green-200 bg-green-50 p-4">
              <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-green-700">
                {t('nextBestProduct')}
              </p>
              <div className="mt-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="flex items-center text-base font-bold text-green-950">
                    <span className="mr-2 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-green-100 text-xs font-black text-green-800">
                      {CATEGORY_INITIALS[outlet.recommended_product.category] ?? 'P'}
                    </span>
                    <span className="truncate">{outlet.recommended_product.product_name}</span>
                  </p>
                  <p className="mt-1 text-sm font-medium text-green-700">{outlet.recommended_product.category}</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="text-[11px] font-semibold text-green-600">{t('expectedUplift')}</p>
                  <p className="text-base font-black text-green-800">
                    Rs {outlet.recommended_product.expected_uplift_inr.toLocaleString('en-IN')}
                  </p>
                </div>
              </div>
              <p className="mt-3 text-sm leading-6 text-green-900">{outlet.recommended_product.justification}</p>
            </div>
          ) : (
            <p className="mb-3 rounded-2xl bg-slate-50 p-4 text-sm italic text-slate-500">{t('noProduct')}</p>
          )}
        </div>

        <div className="shrink-0 border-t border-slate-100 px-4 pb-4 pt-3 sm:px-5">
          <button
            type="button"
            onClick={() => {
              onMarkVisited(outlet);
              onClose();
            }}
            className="w-full rounded-2xl bg-[#0A6E3D] px-4 py-3.5 text-sm font-bold tracking-wide text-white transition hover:bg-[#085c32] active:scale-[0.99]"
          >
            {t('markVisited')}
          </button>
        </div>
      </div>
    </>
  );
}
