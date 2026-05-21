import type { PlanData, PlanMode } from './types';
import { BACKEND_URL, USE_LIVE_BACKEND } from './backendConfig';

const DATE_MAP: Record<PlanMode, string> = {
  today: '2026-02-15',
  post_rain: '2026-02-17',
};

async function loadMockPlan(mode: PlanMode): Promise<PlanData> {
  if (mode === 'today') {
    const data = await import('@/data/mock_plan_today.json');
    return data.default as unknown as PlanData;
  }

  const data = await import('@/data/mock_plan_post_rain.json');
  return data.default as unknown as PlanData;
}

export async function fetchPlan(repId: string, mode: PlanMode): Promise<PlanData> {
  if (!USE_LIVE_BACKEND) {
    return loadMockPlan(mode);
  }

  const date = DATE_MAP[mode];

  try {
    const res = await fetch(
      `${BACKEND_URL}/plan/today?rep_id=${encodeURIComponent(repId)}&date=${date}`,
      {
        cache: 'no-store',
        headers: {
          Accept: 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status} for rep=${repId} date=${date}`);
    }

    const contentType = res.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) {
      throw new Error(`Backend returned non-JSON response: ${contentType || 'unknown content type'}`);
    }

    return res.json() as Promise<PlanData>;
  } catch (error) {
    console.warn(`Live backend unavailable; using ${mode} mock plan.`, error);
    return loadMockPlan(mode);
  }
}
