"""
route_solver.py -- Google OR-Tools VRP solver for the rep's daily route.

Single-vehicle TSP variant. Depot is the centroid of the selected outlets
(closest practical proxy for "rep starting point" when we don't have the
rep's home base). Distance matrix is Haversine in meters, integer-rounded.

CLI:
    python route_solver.py --date 2026-02-15 --rep-id REP_0001

Importable:
    from route_solver import solve_route
    order_indices, total_distance_m = solve_route(outlets_df)

Honest scoping (folded into MODELS.md):
- Distance is great-circle Haversine, not road-network. OSRM hook is left as
  a TODO; the contract field route_polyline is already in lat/lng order so a
  road-network upgrade is a transparent swap-in.
- Depot defaults to outlets-centroid; pass depot_lat/depot_lng to fix it.
- Solver returns identity order with inf distance if OR-Tools is missing or
  no solution is found, so the API stays up.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Haversine + distance matrix
# ---------------------------------------------------------------------------
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000.0  # mean Earth radius, meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_distance_matrix(coords: list[tuple[float, float]]) -> list[list[int]]:
    n = len(coords)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(round(haversine_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])))
            M[i][j] = d
            M[j][i] = d
    return M


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_route(
    outlets: pd.DataFrame,
    depot_lat: Optional[float] = None,
    depot_lng: Optional[float] = None,
    time_limit_s: int = 5,
) -> tuple[list[int], float]:
    """
    Order outlets into an optimal visit sequence.

    Args:
        outlets: DataFrame with `lat`, `lng` columns. Index is irrelevant; the
            returned order references 0..N-1 positions into the passed-in
            DataFrame.
        depot_lat / depot_lng: optional fixed depot. Defaults to centroid of
            outlets' lat/lng.
        time_limit_s: solver wall-clock budget. For N<=10 this is overkill.

    Returns:
        (order_indices, total_distance_m):
            order_indices: positions 0..N-1 in optimal visit sequence
                (depot is excluded).
            total_distance_m: total route distance in meters (depot -> all
                outlets -> back to depot).

    Falls back to identity order with inf distance if OR-Tools is missing
    or the solver returns no solution.
    """
    n = len(outlets)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as e:
        print(f"[route_solver] OR-Tools unavailable: {e}; identity-order fallback")
        return list(range(n)), float("inf")

    lats = outlets["lat"].astype(float).tolist()
    lngs = outlets["lng"].astype(float).tolist()
    if depot_lat is None or depot_lng is None:
        depot_lat = sum(lats) / n
        depot_lng = sum(lngs) / n

    # Index 0 = depot; indices 1..N = outlets in their DataFrame order
    coords = [(float(depot_lat), float(depot_lng))] + list(zip(lats, lngs))
    dist = build_distance_matrix(coords)

    manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)  # 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return dist[f][t]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(sp)
    if solution is None:
        print("[route_solver] no solution; identity-order fallback")
        return list(range(n)), float("inf")

    # Walk the solved tour
    route_nodes: list[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route_nodes.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    # route_nodes starts with depot (0). Closing depot is implicit.

    # Convert coord-space (depot=0, outlets=1..N) -> outlets-space (0..N-1)
    order = [node - 1 for node in route_nodes if node != 0]
    total_m = float(solution.ObjectiveValue())
    return order, total_m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-02-15")
    p.add_argument("--rep-id", default="REP_0001")
    p.add_argument("--topn", type=int, default=10)
    args = p.parse_args()

    from features import get_dataset_root  # local import; avoids cost on bare import

    DATA = Path(__file__).resolve().parent / "data"
    df = pd.read_parquet(DATA / f"features_{args.date}.parquet")
    reps = pd.read_csv(get_dataset_root() / "reps_territory.csv")
    rep_row = reps[reps["rep_id"] == args.rep_id].iloc[0]
    territory_id = rep_row["territory_id"]

    outlets = df[df["territory_id"] == territory_id].copy()
    if len(outlets) > args.topn:
        # Use the trained ranker if available, otherwise just take head()
        try:
            from priority_model import score, is_trained
            if is_trained():
                outlets["_priority_score"] = score(outlets).values
                outlets = outlets.sort_values("_priority_score", ascending=False).head(args.topn)
        except Exception:
            outlets = outlets.head(args.topn)
    outlets = outlets.reset_index(drop=True)
    print(f"[route_solver] rep={args.rep_id} territory={territory_id} outlets={len(outlets)}")

    order, total_m = solve_route(outlets)
    print(f"[route_solver] total distance: {total_m / 1000:.2f} km")
    print()
    print("Route order:")
    for pos, i in enumerate(order, 1):
        r = outlets.iloc[i]
        print(f"  {pos:>2}. {r['retailer_id']} {r['name']:<30} ({float(r['lat']):.4f}, {float(r['lng']):.4f})")


if __name__ == "__main__":
    main()