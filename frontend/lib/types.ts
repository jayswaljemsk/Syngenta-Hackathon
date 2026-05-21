// lib/types.ts
// Source of truth: CONTRACT.md §4.1
// DO NOT modify schema after 19 May 08:30 IST (locked)

export interface RecommendedProduct {
  product_id: string;
  product_name: string;
  category: string;
  expected_uplift_inr: number;
  justification: string;
}

export interface Outlet {
  outlet_id: string;
  name: string;
  address: string;
  lat: number;
  lng: number;
  priority_score: number;
  confidence: 'High' | 'Medium' | 'Low';
  route_position: number;
  reason_strings: string[];
  recommended_product: RecommendedProduct | null;
  estimated_visit_minutes: number;
}

export interface Anomaly {
  type: string;
  severity: 'high' | 'medium' | 'low' | 'info';
  description: string;
  affected_outlets: string[];
}

export interface PlanData {
  rep_id: string;
  date: string;
  territory: string;
  synced_at: string;
  weather_summary: string;
  outlets: Outlet[];
  route_polyline: [number, number][];
  anomalies: Anomaly[];
}

export type PlanMode = 'today' | 'post_rain';
export type Language = 'en' | 'hi' | 'mr';
