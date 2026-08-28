from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, Iterable

import math
import re
import sympy as sp


# Keep consistent with Network.update_graphs mapping
_UPDATE_ALIASES = {
    # Lens-like / user-friendly strings:
    "NEVER": None,                   # allow, but maps to default; user can still manual export
    "EXAMPLES": 0,
    "EXAMPLE": 0,

    "WEIGHT_UPDATES": 1,
    "WEIGHT_UPDATE": 1,

    "COMPLETION_OF_A_BATCH": 2,
    "BATCH": 2,
    "BATCHES": 2,

    "PROGRESS_REPORTS": 3,
    "PROGRESS_REPORT": 3,

    "TRAINING_AND_TESTING": 4,
}

def _normalize_update_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().upper())


def parse_update_after(update: Union[str, int, None]) -> Optional[int]:
    """Convert update strings to update_after enum (0..4). Returns int or None (NEVER)."""
    if update is None:
        return 3  # default = progress report
    if isinstance(update, int):
        if update < 0 or update > 4:
            raise ValueError("update_after int must be in [0,4]")
        return update

    key = _normalize_update_name(update)

    if key in _UPDATE_ALIASES:
        return _UPDATE_ALIASES[key]

    # unique prefix match
    candidates = {k: v for k, v in _UPDATE_ALIASES.items() if k.startswith(key)}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if len(candidates) == 0:
        raise ValueError(f"Unrecognized updates value: {update!r}")
    raise ValueError(f"Ambiguous updates value {update!r}; matches {sorted(candidates)}")


# -----------------------------------------------------------------------------
# Headless data structures
# -----------------------------------------------------------------------------


@dataclass
class HeadlessTrace:
    """A single series within a graph."""
    expr: Optional[sp.Expr] = None
    label: str = ""
    active: bool = True

    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    # stored snapshots (each is a y-series; x stored at graph level)
    stored: List[List[float]] = field(default_factory=list)

    def clear(self) -> None:
        self.x.clear()
        self.y.clear()

    def store(self) -> None:
        if self.y:
            self.stored.append(self.y[:])
        self.clear()


@dataclass
class HeadlessGraph:
    """
    Headless graph with a small compatibility layer:
      - GUI graphs have plot_variable / plot_data / x_data lists.
      - Headless graphs support multiple traces via `traces`.
    """
    update_after: int = 3
    special_variables: Dict[str, str] = field(default_factory=dict)
    window_name: str = "HeadlessGraph"

    traces: List[Optional[HeadlessTrace]] = field(default_factory=list)

    # Fields Network may read/write
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = -1.0
    y_max: float = 1.0
    min_x_data: float = 0.0
    graph_clock: float = 0.0

    # Stored snapshots at graph-level (for parity with Lens "graph store")
    stored_traces: List[List[List[float]]] = field(default_factory=list)  # list of trace-y lists
    stored_x: List[List[float]] = field(default_factory=list)

    # --- GUI-compat shim properties (map to first active trace) ---

    @property
    def plot_variable(self) -> Optional[sp.Expr]:
        t = self._first_active_trace()
        return t.expr if t else None

    @plot_variable.setter
    def plot_variable(self, value: Any) -> None:
        if not self.traces:
            self.traces.append(HeadlessTrace(expr=None, label=""))

        expr = sp.sympify(value) if value is not None else None

        for t in self.traces:
            if t is not None and t.active:
                t.expr = expr
                t.label = str(value)
                return

        for t in self.traces:
            if t is not None:
                t.expr = expr
                t.label = str(value)
                t.active = True
                return

    @property
    def plot_data(self) -> List[List[float]]:
        return [t.y for t in self.traces if t is not None]

    @property
    def x_data(self) -> List[List[float]]:
        return [t.x for t in self.traces if t is not None]

    # --- methods called by Network.update_graphs ---

    def update_clock(self, delta: float) -> None:
        self.graph_clock += float(delta)

    def set_clock(self, value: float) -> None:
        self.graph_clock = float(value)

    def update_x_data(self, x: float) -> None:
        """Append x to all active traces."""
        xf = float(x)
        for t in self._active_traces():
            t.x.append(xf)

        first = self._first_active_trace()
        if first and first.x:
            self.min_x_data = float(first.x[0])

    def update_trace(self) -> None:
        # In headless, trace boundaries are explicit via store/clear/restart.
        return

    def update_xy_limits(self) -> None:
        """Auto-range over recent active y values."""
        ys: List[float] = []
        for t in self._active_traces():
            ys.extend(t.y[-100:])
        if not ys:
            return
        y0 = min(ys)
        y1 = max(ys)
        if not (math.isfinite(y0) and math.isfinite(y1)):
            return
        pad = 0.05 * (y1 - y0) if y1 != y0 else 0.5
        self.y_min = y0 - pad
        self.y_max = y1 + pad

    # --- headless-specific convenience ---

    def clear(self) -> None:
        for t in self.traces:
            if t is not None:
                t.clear()
        self.graph_clock = 0.0
        self.min_x_data = 0.0

    def store(self) -> None:
        snap: List[List[float]] = []
        first_x: Optional[List[float]] = None
        for t in self._active_traces():
            snap.append(t.y[:])
            if first_x is None:
                first_x = t.x[:]
            t.clear()

        if snap and first_x is not None:
            self.stored_traces.append(snap)
            self.stored_x.append(first_x)

        self.graph_clock = 0.0
        self.min_x_data = 0.0

    def restart(self) -> None:
        # match the spirit of GUI restart
        self.store()

    # --- internals ---

    def _active_traces(self) -> Iterable[HeadlessTrace]:
        for t in self.traces:
            if t is not None and t.active:
                yield t

    def _first_active_trace(self) -> Optional[HeadlessTrace]:
        for t in self._active_traces():
            return t
        for t in self.traces:
            if t is not None:
                return t
        return None


# -----------------------------------------------------------------------------
# Lens-like wrapper functions
# -----------------------------------------------------------------------------


def _ensure_graphs_list(net) -> None:
    if not hasattr(net, "graphs") or net.graphs is None:
        net.graphs = []


def _parse_id_list(arg: Union[str, int, Sequence[int], None], *, max_len: int) -> List[int]:
    if arg is None:
        return []
    if isinstance(arg, int):
        return [arg]
    if isinstance(arg, str):
        if arg.strip() == "*":
            return list(range(max_len))
        parts = [p for p in re.split(r"[\s,]+", arg.strip()) if p]
        return [int(p) for p in parts]
    return [int(x) for x in arg]


def graphObject(
    net,
    objects: Union[str, Sequence[str], None] = None,
    *,
    updates: Union[str, int, None] = "PROGRESS_REPORTS",
) -> List[int]:
    """Create one headless graph per object expression. Returns list of graph IDs."""
    _ensure_graphs_list(net)

    if objects is None:
        objects_list = ["error"]
    elif isinstance(objects, str):
        objects_list = [o for o in objects.split() if o.strip()]
    else:
        objects_list = list(objects)

    update_after = parse_update_after(updates)
    if update_after is None:
        update_after = 3

    gids: List[int] = []
    for obj in objects_list:
        g = HeadlessGraph(
            update_after=update_after,
            special_variables={},
            window_name=f"HeadlessGraph[{obj}]",
        )
        g.traces.append(HeadlessTrace(expr=sp.sympify(obj), label=str(obj), active=True))
        net.graphs.append(g)
        gids.append(len(net.graphs) - 1)

    return gids


def graph(net, action: str, graph_list: Union[str, int, Sequence[int], None] = None) -> Union[int, List[int], None]:
    """Create/list/delete/update/store/clear graphs. Leaves None holes on delete for stable IDs."""
    _ensure_graphs_list(net)
    action = action.strip().lower()

    if action == "create":
        g = HeadlessGraph(window_name="HeadlessGraph")
        g.traces.append(HeadlessTrace(expr=None, label="trace0", active=True))
        net.graphs.append(g)
        return len(net.graphs) - 1

    if action == "list":
        return [i for i, g in enumerate(net.graphs) if g is not None]

    gids = _parse_id_list(graph_list, max_len=len(net.graphs))

    if action == "delete":
        for gid in gids:
            if 0 <= gid < len(net.graphs):
                net.graphs[gid] = None
        return None

    for gid in gids:
        if not (0 <= gid < len(net.graphs)):
            continue
        g = net.graphs[gid]
        if g is None:
            continue

        if action in {"refresh", "hide", "show"}:
            continue
        if action == "update":
            g.update_clock(1)
            if hasattr(net, "send_data_to_graph_viewer"):
                net.send_data_to_graph_viewer(g)
            continue
        if action == "store":
            g.store() if hasattr(g, "store") else None
            continue
        if action == "clear":
            g.clear() if hasattr(g, "clear") else None
            continue

        raise ValueError(f"Unsupported graph action: {action!r}")

    return None


def trace(
    net,
    action: str,
    graph_id: int,
    obj: Optional[str] = None,
    trace_list: Union[str, int, Sequence[int], None] = None,
) -> Union[int, List[int], None]:
    """Create/list/delete/store/clear traces in a headless graph."""
    _ensure_graphs_list(net)
    action = action.strip().lower()

    if not (0 <= graph_id < len(net.graphs)) or net.graphs[graph_id] is None:
        raise ValueError(f"Invalid graph id: {graph_id}")

    g = net.graphs[graph_id]
    if not hasattr(g, "traces"):
        raise ValueError("trace() is only supported for HeadlessGraph graphs")

    if action == "create":
        expr = sp.sympify(obj) if obj else None
        label = str(obj) if obj else f"trace{len(g.traces)}"
        g.traces.append(HeadlessTrace(expr=expr, label=label, active=True))
        return len(g.traces) - 1

    if action == "list":
        return [i for i, t in enumerate(g.traces) if t is not None]

    tids = _parse_id_list(trace_list, max_len=len(g.traces))

    if action == "delete":
        for tid in tids:
            if 0 <= tid < len(g.traces):
                g.traces[tid] = None
        return None

    for tid in tids:
        if not (0 <= tid < len(g.traces)):
            continue
        t = g.traces[tid]
        if t is None:
            continue
        if action == "store":
            t.store()
            continue
        if action == "clear":
            t.clear()
            continue
        raise ValueError(f"Unsupported trace action: {action!r}")

    return None


def exportGraph(
    net,
    gid: int,
    filename: str,
    *,
    labels: bool = False,
    gnuplot: bool = False,
) -> None:
    """Export active traces to a TSV (default) or gnuplot blocks."""
    _ensure_graphs_list(net)
    if not (0 <= gid < len(net.graphs)) or net.graphs[gid] is None:
        raise ValueError(f"Invalid graph id: {gid}")

    g = net.graphs[gid]

    # If someone passes a GUI graph object here, fall back to legacy buffers.
    if not hasattr(g, "traces"):
        traces = getattr(g, "plot_data", [[]])
        xdata = getattr(g, "x_data", [[]])
        x = xdata[-1] if xdata and isinstance(xdata[0], list) else xdata
        _export_xy_matrix(filename, x=x, traces=traces, labels=labels, gnuplot=gnuplot)
        return

    active_traces = [t for t in g.traces if t is not None and t.active]
    if not active_traces:
        active_traces = [t for t in g.traces if t is not None]

    x = active_traces[0].x if active_traces else []
    traces_y = [t.y for t in active_traces]
    trace_labels = [t.label or f"trace{i}" for i, t in enumerate(active_traces)]

    _export_xy_matrix(
        filename,
        x=x,
        traces=traces_y,
        labels=labels,
        gnuplot=gnuplot,
        header_labels=trace_labels,
    )


def _export_xy_matrix(
    filename: str,
    *,
    x: List[float],
    traces: List[List[float]],
    labels: bool,
    gnuplot: bool,
    header_labels: Optional[List[str]] = None,
) -> None:
    def safe_get(seq, i):
        return seq[i] if i < len(seq) else ""

    max_len = max((len(t) for t in traces), default=0)

    with open(filename, "w", encoding="utf-8") as f:
        if gnuplot:
            for tr in traces:
                for i in range(len(tr)):
                    f.write(f"{safe_get(x, i)}\t{tr[i]}\n")
                f.write("\n\n")
            return

        if labels:
            cols = header_labels if header_labels is not None else [f"trace{tid}" for tid in range(len(traces))]
            f.write("\t".join(["x"] + cols) + "\n")

        for i in range(max_len):
            row = [safe_get(x, i)] + [safe_get(tr, i) for tr in traces]
            f.write("\t".join(str(v) for v in row) + "\n")