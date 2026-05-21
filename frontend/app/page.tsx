// app/page.tsx
//
// Server component — pre-loads both mock plans at build time
// (or from Krishna's API on Day 2 by changing fetchPlan.ts).
//
// This file never changes. All interactivity lives in PlanView.

import PlanView from '@/components/PlanView';
import { fetchPlan } from '@/lib/fetchPlan';

export const dynamic = 'force-dynamic';

export default async function Home() {
  // Both plans fetched in parallel at request time.
  // On Day 1: reads local mock JSON (instant, no network).
  // On Day 2: flip USE_LIVE_BACKEND=true in lib/fetchPlan.ts.
  const [todayPlan, rainPlan] = await Promise.all([
    fetchPlan('REP_0001', 'today'),
    fetchPlan('REP_0001', 'post_rain'),
  ]);

  return <PlanView todayPlan={todayPlan} rainPlan={rainPlan} />;
}
