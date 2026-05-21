'use client';

import 'leaflet/dist/leaflet.css';
import { useEffect, useRef } from 'react';
import type { Map as LMap, LayerGroup } from 'leaflet';
import type { PlanData, Outlet } from '@/lib/types';

function pinColor(pos: number): string {
  if (pos <= 3) return '#DC2626';
  if (pos <= 7) return '#D97706';
  return '#16A34A';
}

function makePinSvg(pos: number, color: string, state: 'normal' | 'selected' | 'visited'): string {
  const fill = state === 'visited' ? '#94A3B8' : color;
  const text = state === 'visited' ? '#64748B' : color;
  const opacity = state === 'visited' ? '0.72' : '1';
  const stroke = state === 'selected' ? '#111827' : 'white';
  const label = state === 'visited' ? 'OK' : String(pos);

  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="46" viewBox="0 0 36 46">
      <defs>
        <filter id="ds${pos}" x="-30%" y="-20%" width="160%" height="160%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="1.5" flood-opacity="0.28"/>
        </filter>
      </defs>
      <path
        d="M18 2C10.27 2 4 8.27 4 16c0 10.5 14 28 14 28S32 26.5 32 16C32 8.27 25.73 2 18 2z"
        fill="${fill}" filter="url(#ds${pos})" opacity="${opacity}"
      />
      <circle cx="18" cy="16" r="10.5" fill="white" fill-opacity="0.97" stroke="${stroke}" stroke-width="${state === 'selected' ? '2' : '0'}"/>
      <text
        x="18" y="20.5"
        text-anchor="middle"
        font-family="system-ui,-apple-system,sans-serif"
        font-size="${label.length > 1 ? 8 : 10.5}"
        font-weight="700"
        fill="${text}"
      >${label}</text>
    </svg>`;
}

function renderPlan(
  L: typeof import('leaflet'),
  plan: PlanData,
  markersLayer: LayerGroup,
  polylineLayer: LayerGroup,
  onSelect: (outlet: Outlet) => void,
  visitedIds: Set<string>,
  selectedOutletId: string | null,
): void {
  markersLayer.clearLayers();
  polylineLayer.clearLayers();

  L.polyline(plan.route_polyline, {
    color: '#0A6E3D',
    weight: 2.5,
    opacity: 0.55,
    dashArray: '7 5',
  }).addTo(polylineLayer);

  plan.outlets.forEach((outlet) => {
    const color = pinColor(outlet.route_position);
    const state = visitedIds.has(outlet.outlet_id)
      ? 'visited'
      : selectedOutletId === outlet.outlet_id
        ? 'selected'
        : 'normal';
    const icon = L.divIcon({
      html: makePinSvg(outlet.route_position, color, state),
      className: '',
      iconSize: [36, 46],
      iconAnchor: [18, 46],
    });
    const marker = L.marker([outlet.lat, outlet.lng], { icon });
    marker.on('click', () => onSelect(outlet));
    markersLayer.addLayer(marker);
  });
}

interface MapViewProps {
  plan: PlanData;
  onOutletSelect: (outlet: Outlet) => void;
  visitedIds: Set<string>;
  selectedOutletId: string | null;
}

export default function MapView({ plan, onOutletSelect, visitedIds, selectedOutletId }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LMap | null>(null);
  const markersRef = useRef<LayerGroup | null>(null);
  const polylineRef = useRef<LayerGroup | null>(null);
  const leafletRef = useRef<typeof import('leaflet') | null>(null);

  const onSelectRef = useRef(onOutletSelect);
  useEffect(() => {
    onSelectRef.current = onOutletSelect;
  }, [onOutletSelect]);

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;

    import('leaflet').then((L) => {
      if (mapRef.current) return;

      const map = L.map(containerRef.current!, {
        zoomControl: true,
        scrollWheelZoom: false,
      }).setView([23.20, 77.08], 7);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'OpenStreetMap',
        maxZoom: 17,
      }).addTo(map);

      const markersLayer = L.layerGroup().addTo(map);
      const polylineLayer = L.layerGroup().addTo(map);

      leafletRef.current = L;
      mapRef.current = map;
      markersRef.current = markersLayer;
      polylineRef.current = polylineLayer;

      renderPlan(L, plan, markersLayer, polylineLayer, (outlet) => onSelectRef.current(outlet), visitedIds, selectedOutletId);
      map.fitBounds(plan.route_polyline, { padding: [28, 28], maxZoom: 10 });
    });

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
      markersRef.current = null;
      polylineRef.current = null;
      leafletRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const L = leafletRef.current;
    if (!L || !markersRef.current || !polylineRef.current || !mapRef.current) return;
    renderPlan(L, plan, markersRef.current, polylineRef.current, (outlet) => onSelectRef.current(outlet), visitedIds, selectedOutletId);
    mapRef.current.fitBounds(plan.route_polyline, {
      padding: [28, 28],
      maxZoom: 10,
      animate: true,
      duration: 0.55,
    });
  }, [plan, visitedIds, selectedOutletId]);

  return <div ref={containerRef} className="h-full min-h-[420px] w-full sm:min-h-0" />;
}
