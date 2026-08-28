"""Unit viewer layout builder (Lens-style ``plotRow``).

This module isolates *layout construction* from Tk rendering.

The GUI (unit viewer) renders the layout grid stored on the network as
``net.plot_layout``. A layout is:

    layout: list[list[Cell]]

Where each Cell is either:

- ``UnitCell(group_name, unit_index)``
- ``BlankCell()``

The Lens ``plotRow`` language is stateful:
- per-group cursors remember where ``next`` continues
- a global set remembers which units have been plotted already
- ``unit`` can re-plot a unit and blanks its previous location

This module provides:

- ``PlotRowState``: explicit state (can be stored on the network)
- ``reset_plot_state(net)``
- ``apply_plotrow(net, ... )`` / ``apply_plotrow_from_command(net, ...)``
- ``build_layout(net, commands, ...)`` convenience helper

No GUI code is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Set


# Cells
@dataclass(frozen=True)
class UnitCell:
    """A single plotted unit in a plot layout grid."""

    group_name: str
    unit_index: int


@dataclass(frozen=True)
class BlankCell:
    """A blank cell placeholder in a plot layout grid."""

    pass


Cell = Union[UnitCell, BlankCell]
Layout = List[List[Cell]]


# State
@dataclass
class PlotRowState:
    """State carried across multiple ``plotRow`` calls."""

    cursor: Dict[str, int] = field(default_factory=dict)
    plotted: Set[Tuple[str, int]] = field(default_factory=set)
    # Maps (group, idx) -> (row_i, col_i) for 'unit' replot blanking
    unit_pos: Dict[Tuple[str, int], Tuple[int, int]] = field(default_factory=dict)


# Tokenization helpers
def tokenize_plot_cmd(cmd: Union[str, Iterable[str], None]) -> List[str]:
    """Tokenize a plotRow command.

    Accepts:
      - iterable of strings

    Returns tokens with the leading "plotRow" removed if present.
    """

    if cmd is None:
        return []

    if isinstance(cmd, str):
        tokens = cmd.strip().split()
    else:
        tokens = [str(x) for x in cmd]

    if tokens and tokens[0] == "plotRow":
        tokens = tokens[1:]
    return tokens


def _maybe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# Public API
def reset_plot_state(net) -> PlotRowState:
    """Reset any plotRow-related state on the network.

    This also clears ``net.plot_layout``.
    """

    state = PlotRowState()
    setattr(net, "plot_layout", [])
    setattr(net, "_plotrow_state", state)
    return state


def get_plot_state(net, *, create: bool = True) -> PlotRowState:
    """Get or create PlotRowState stored on ``net._plotrow_state``."""

    st = getattr(net, "_plotrow_state", None)
    if st is None and create:
        st = PlotRowState()
        setattr(net, "_plotrow_state", st)
    return st

def clear_plot_layout(net) -> None:
    """Remove plotRow-produced layout + state from the network."""
    for attr in ("plot_layout", "_plotrow_state"):
        if hasattr(net, attr):
            try:
                delattr(net, attr)
            except Exception:
                setattr(net, attr, None)


def apply_plotrow_from_command(
    net,
    cmd: Union[str, Iterable[str]],
    *,
    layout: Optional[Layout] = None,
    state: Optional[PlotRowState] = None,
    plotCol: Optional[int] = None,
) -> Tuple[Layout, PlotRowState]:
    """Apply a single raw ``plotRow ...`` command and return (layout, state)."""

    tokens = tokenize_plot_cmd(cmd)
    if plotCol is not None:
        setattr(net, "plotCol", int(plotCol))
    if not tokens:
        return apply_plotrow(net, 1, [], layout=layout, state=state)

    num_rows: Union[int, str] = 1
    if tokens and (tokens[0].isdigit() or tokens[0] == "*"):
        num_rows = tokens[0]
        tokens = tokens[1:]

    return apply_plotrow(net, num_rows, tokens, layout=layout, state=state)



def build_layout(
    net,
    *commands: Union[str, Iterable[str], Sequence[Union[str, Iterable[str]]]],
    reset: bool = True,
    plotCol: Optional[int] = None,
    copy: bool = False,
) -> Tuple[Layout, PlotRowState]:
    """
    Build and return (layout, state) by applying one or more plotRow commands.

    Accepts:
      - varargs: build_layout(net, "plotRow ...", "plotRow ...", ...)
      - a single list/tuple of commands:
            build_layout(net, ["plotRow ...", "plotRow ..."])
      - commands can be strings or token iterables.

    Parameters
    ----------
    reset:
        If True, clears previous plotRow state/layout before applying commands.
    plotCol:
        Optional net.plotCol override (enables 'fill' expansion).
    copy:
        If True, returns a shallow copy of the layout grid (rows are copied).
        Useful if the caller wants to reuse the returned layout without being
        affected by later mutations.

    Returns
    -------
    (layout, state)
    """
    if len(commands) == 1 and isinstance(commands[0], (list, tuple)):
        cmds = list(commands[0])  # list of commands
    else:
        cmds = list(commands)

    state = reset_plot_state(net) if reset else get_plot_state(net, create=True)
    layout: Layout = [] if reset else (getattr(net, "plot_layout", None) or [])
    setattr(net, "plot_layout", layout)
    setattr(net, "_plotrow_state", state)

    if plotCol is not None:
        setattr(net, "plotCol", int(plotCol))

    for cmd in cmds:
        layout, state = apply_plotrow_from_command(net, cmd, layout=layout, state=state)

    if copy:
        layout_out = [list(r) for r in layout]
        return layout_out, state

    return layout, state

# Core implementation
def apply_plotrow(
    net,
    num_rows: Union[int, str],
    tokens: Sequence[str],
    *,
    layout: Optional[Layout] = None,
    state: Optional[PlotRowState] = None,
) -> Tuple[Layout, PlotRowState]:
    """Apply a plotRow specification.

    Parameters
    ----------
    net:
        Network-like object. Must have a ``groups`` iterable; each group should
        have ``name`` and ``num_units``.

    num_rows:
        int or "*". "*" repeats rows until a row produces 0 plotted units.

    tokens:
        the plotRow block tokens (already without leading "plotRow" and without
        the optional numRows token).

    layout/state:
        Provide to extend an existing layout; otherwise we use net.plot_layout
        and net._plotrow_state.

    Returns
    -------
    (layout, state)
    """

    if layout is None:
        layout = getattr(net, "plot_layout", None) or []
    if state is None:
        state = get_plot_state(net, create=True)

    setattr(net, "plot_layout", layout)
    setattr(net, "_plotrow_state", state)

    repeat_forever = (isinstance(num_rows, str) and num_rows.strip() == "*")
    rows_target = 1 if repeat_forever else int(num_rows)

    def _get_group(name: str):
        if hasattr(net, "get_group_by_name"):
            try:
                return net.get_group_by_name(name)
            except Exception:
                pass
        for g in getattr(net, "groups", []):
            if getattr(g, "name", None) == name:
                return g
        return None

    def _total_units(group_name: str) -> int:
        g = _get_group(group_name)
        return int(getattr(g, "num_units", 0) or 0) if g is not None else 0

    def _place_cells(row_cells: List[Cell], new_cells: List[Cell], row_i: int) -> None:
        for cell in new_cells:
            col_i = len(row_cells)
            row_cells.append(cell)
            if isinstance(cell, UnitCell):
                state.unit_pos[(cell.group_name, cell.unit_index)] = (row_i, col_i)

    def _take_next_unused(group_name: str, n: int) -> List[UnitCell]:
        total = _total_units(group_name)
        cursor = int(state.cursor.get(group_name, 0))
        out: List[UnitCell] = []
        taken = 0
        while cursor < total and taken < n:
            key = (group_name, cursor)
            if key not in state.plotted:
                out.append(UnitCell(group_name, cursor))
                state.plotted.add(key)
                taken += 1
            cursor += 1
        state.cursor[group_name] = cursor
        return out

    def _make_fixed_block(justify: str, group_name: str, width: int) -> List[Cell]:
        width = max(0, int(width))
        units = _take_next_unused(group_name, width)
        k = len(units)
        blanks: List[Cell] = [BlankCell()] * (width - k)

        if justify == "left":
            return list(units) + blanks
        if justify == "right":
            return blanks + list(units)
        if justify == "center":
            left_pad = (width - k) // 2
            right_pad = (width - k) - left_pad
            return ([BlankCell()] * left_pad) + list(units) + ([BlankCell()] * right_pad)
        return list(units) + blanks

    def _blank_old_position_if_any(group_name: str, idx: int) -> None:
        key = (group_name, idx)
        if key not in state.unit_pos:
            return
        r, c = state.unit_pos[key]
        if 0 <= r < len(layout):
            row = layout[r]
            if 0 <= c < len(row):
                row[c] = BlankCell()

    def _parse_unit_spec(tok: str, maybe_idx: Optional[str]) -> Tuple[Optional[str], Optional[int], int]:
        """Return (group, idx, consumed_tokens_after_cmd)."""
        if tok is None:
            return None, None, 0
        if ":" in tok:
            g, i = tok.split(":", 1)
            try:
                return g, int(i), 1
            except Exception:
                return g, None, 1
        # two-token form: group idx
        if maybe_idx is None:
            return tok, None, 1
        try:
            return tok, int(maybe_idx), 2
        except Exception:
            return tok, None, 2

    def _parse_one_row(tokens_for_row: Sequence[str], row_i: int) -> Tuple[List[Cell], int]:
        row_cells: List[Cell] = []
        units_plotted = 0
        fill_positions: List[int] = []
        i = 0

        def _normalize_cmd(x: str) -> str:
            if not x:
                return ""
            x = x.strip()
            if x in (
                "next",
                "lnext",
                "cnext",
                "rnext",
                "span",
                "unit",
                "blank",
                "fill",
            ):
                return x
            # Lens abbreviations: first char
            return x[0]

        while i < len(tokens_for_row):
            cmd = _normalize_cmd(tokens_for_row[i])

            # next <group> <n>
            if cmd in ("next", "n"):
                if i + 2 >= len(tokens_for_row):
                    break
                gname = tokens_for_row[i + 1]
                n = _maybe_int(tokens_for_row[i + 2], 0)
                n = max(0, n)

                cells = _take_next_unused(gname, n)

                # >>> FIX: keep alignment in '*' (repeat) mode by padding to width n
                if repeat_forever and len(cells) < n:
                    padded: List[Cell] = list(cells) + [BlankCell()] * (n - len(cells))
                    _place_cells(row_cells, padded, row_i)
                    units_plotted += len(cells)  # only count actual units
                else:
                    _place_cells(row_cells, list(cells), row_i)
                    units_plotted += len(cells)

                i += 3
                continue

            # lnext/cnext/rnext <group> <n>
            if cmd in ("lnext", "l", "cnext", "c", "rnext", "r"):
                if i + 2 >= len(tokens_for_row):
                    break
                gname = tokens_for_row[i + 1]
                n = _maybe_int(tokens_for_row[i + 2], 0)

                if cmd in ("lnext", "l"):
                    block = _make_fixed_block("left", gname, n)
                elif cmd in ("cnext", "c"):
                    block = _make_fixed_block("center", gname, n)
                else:
                    block = _make_fixed_block("right", gname, n)

                _place_cells(row_cells, list(block), row_i)
                units_plotted += sum(1 for x in block if isinstance(x, UnitCell))
                i += 3
                continue

            # span <group> <start> <n>
            if cmd in ("span", "s"):
                if i + 3 >= len(tokens_for_row):
                    break
                gname = tokens_for_row[i + 1]
                start = _maybe_int(tokens_for_row[i + 2], 0)
                n = _maybe_int(tokens_for_row[i + 3], 0)
                total = _total_units(gname)

                block: List[Cell] = []
                for off in range(max(0, n)):
                    idx = start + off
                    if idx < 0 or idx >= total:
                        block.append(BlankCell())
                        continue
                    key = (gname, idx)
                    if key in state.plotted:
                        block.append(BlankCell())
                    else:
                        cell = UnitCell(gname, idx)
                        block.append(cell)
                        state.plotted.add(key)
                        units_plotted += 1

                _place_cells(row_cells, block, row_i)
                i += 4
                continue

            # unit <unit>
            if cmd in ("unit", "u"):
                if i + 1 >= len(tokens_for_row):
                    break
                gname, idx, consumed = _parse_unit_spec(
                    tokens_for_row[i + 1],
                    tokens_for_row[i + 2] if i + 2 < len(tokens_for_row) else None,
                )
                if gname is None or idx is None:
                    i += 1 + consumed
                    continue

                _blank_old_position_if_any(gname, idx)
                state.plotted.add((gname, idx))

                _place_cells(row_cells, [UnitCell(gname, idx)], row_i)
                units_plotted += 1
                i += 1 + consumed
                continue

            # blank <n>
            if cmd in ("blank", "b"):
                if i + 1 >= len(tokens_for_row):
                    break
                n = _maybe_int(tokens_for_row[i + 1], 0)
                _place_cells(row_cells, [BlankCell()] * max(0, n), row_i)
                i += 2
                continue

            # fill
            if cmd in ("fill", "f"):
                fill_positions.append(len(row_cells))
                i += 1
                continue

            # unknown
            i += 1

        # Expand fill blocks using net.plotCol if present
        plot_col = getattr(net, "plotCol", None)
        if plot_col is not None and fill_positions:
            try:
                plot_col = int(plot_col)
            except Exception:
                plot_col = None

        if plot_col is not None and fill_positions:
            fixed_w = len(row_cells)
            remaining = plot_col - fixed_w
            if remaining > 0:
                k = len(fill_positions)
                base = remaining // k
                extra = remaining % k

                # Insert from right to left to keep indices valid
                for j in range(k - 1, -1, -1):
                    pos = fill_positions[j]
                    add = base + (1 if j < extra else 0)
                    if add <= 0:
                        continue
                    row_cells[pos:pos] = [BlankCell()] * add

                    # Shift unit_pos for cells in this row
                    for key, (rr, cc) in list(state.unit_pos.items()):
                        if rr == row_i and cc >= pos:
                            state.unit_pos[key] = (rr, cc + add)

        return row_cells, units_plotted


    rows_added = 0
    while True:
        if not repeat_forever and rows_added >= rows_target:
            break

        row_i = len(layout)
        row_cells, units_plotted = _parse_one_row(tokens, row_i)
        layout.append(row_cells)
        rows_added += 1

        if repeat_forever and units_plotted == 0:
            layout.pop()  # remove trailing empty row
            break

    return layout, state


def plot_all(
    net,
    group_name: str,
    *,
    layout: Optional[Layout] = None,
    state: Optional[PlotRowState] = None,
    plotCol: Optional[int] = None,
) -> Tuple[Layout, PlotRowState]:
    """
    Lens-style: plotAll <group>

    Plots all *unplotted* units in `group_name` in as many rows as needed.
    Row width priority:
      1) group.numColumns (if > 0)
      2) net.plotCol (if set and > 0)
      3) plotCol argument (if provided and > 0)
      4) fallback: group.num_units

    Final partial row is centered within the row width.

    Returns (layout, state). Also updates net.plot_layout and net._plotrow_state.
    """
    if layout is None:
        layout = getattr(net, "plot_layout", None) or []
    if state is None:
        state = get_plot_state(net, create=True)

    setattr(net, "plot_layout", layout)
    setattr(net, "_plotrow_state", state)

    # --- resolve group + width ---
    def _get_group(name: str):
        if hasattr(net, "get_group_by_name"):
            try:
                return net.get_group_by_name(name)
            except Exception:
                pass
        for g in getattr(net, "groups", []):
            if getattr(g, "name", None) == name:
                return g
        return None

    g = _get_group(group_name)
    if g is None:
        return layout, state

    group_cols = int(getattr(g, "numColumns", 0) or 0)

    net_cols = getattr(net, "plotCol", None)
    try:
        net_cols = int(net_cols) if net_cols is not None else 0
    except Exception:
        net_cols = 0

    arg_cols = 0
    if plotCol is not None:
        try:
            arg_cols = int(plotCol)
        except Exception:
            arg_cols = 0

    width = group_cols if group_cols > 0 else (net_cols if net_cols > 0 else (arg_cols if arg_cols > 0 else 0))
    if width <= 0:
        width = int(getattr(g, "num_units", 0) or 0)

    # --- repeatedly add rows ---
    while True:
        before = len(state.plotted)

        apply_plotrow(net, 1, ["next", group_name, str(width)], layout=layout, state=state)

        after = len(state.plotted)
        plotted_this_row = after - before
        if plotted_this_row == 0:
            if layout:
                layout.pop()
            break

        if plotted_this_row < width:
            # Center the final partial row in-place.
            last_row_i = len(layout) - 1
            last_row = layout[last_row_i]
            placed = [c for c in last_row if isinstance(c, UnitCell)]

            k = len(placed)
            left_pad = (width - k) // 2
            right_pad = (width - k) - left_pad
            new_row: List[Cell] = [BlankCell()] * left_pad + placed + [BlankCell()] * right_pad

            layout[last_row_i] = new_row
            for col_i, cell in enumerate(new_row):
                if isinstance(cell, UnitCell):
                    state.unit_pos[(cell.group_name, cell.unit_index)] = (last_row_i, col_i)

            break

    return layout, state

