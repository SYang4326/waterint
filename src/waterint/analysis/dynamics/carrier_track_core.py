"""Strict OH- carrier tracking on site-current event tables.

Rules:
  * carrier = water O (event row 0, i.e. not the frozen solid) with nH == 1 (H -> nearest O within the profile cutoff)
  * between consecutive frames, an old carrier (nH 1 -> 2) is mapped to a new carrier (2 -> 1) by the unique shortest
    path of <= 4 changed O-H edges (both owners assigned) from the new to the old carrier whose internal O are water with
    nH 2 in both frames; forced one-to-one matching; two accepted paths sharing an edge or internal O are both rejected
  * carriers present in both frames keep their identity; unmapped old carriers end (ambiguous_relay / surface_exchange /
    assignment_gap / OH_to_O2minus / OH_to_H2O / OH_to_H3Oplus / OH_identity_break), unmapped new carriers start
  * carrier displacement of an edge = donor - acceptor O midpoint vector = -hop of the event
The input is the ``waterint-events-1`` table written by ``site-current`` (events: enabled), so long runs are tracked
without reading the trajectory again; only frames with owner changes are visited. Every sampled site frame the tracked
carrier set is checked against the pass site table (water OH- = row 0, nH 1); any mismatch is an error.
Besides the per-track counters this module records the carrier-O schedule (track, frame, O id) so that the
per-frame carrier site can be rebuilt from positions (``carrier_msd``).
Region codes (lattice, L1 [1.5, 2.8), L2 [2.8, 4.0), bulk [4.0, 30), below L1, above) at the event
z_rel (midpoint of the two frames); initial carriers have region -1.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

MAX_RELAY = 4
REGIONS = ["lattice", "L1", "L2", "bulk", "other_below_L1", "other_above_bulk"]


def region(row, z):
    if row != 0:
        return 0
    if z < 1.5:
        return 4
    if z < 2.8:
        return 1
    if z < 4.0:
        return 2
    if z < 30.0:
        return 3
    return 5


def enumerate_paths(start, goal, adjacency, prev, cur, lattice):
    found = []
    queue = [(start, [start], [])]
    while queue:
        node, nodes, edges = queue.pop(0)
        if len(edges) >= MAX_RELAY:
            continue
        for nxt, h, eid in adjacency.get(node, []):
            if nxt in nodes:
                continue
            new_edges = edges + [(node, nxt, h)]
            if nxt == goal:
                found.append(new_edges)
                continue
            if lattice.get(nxt, False) or prev[nxt] != 2 or cur[nxt] != 2:
                continue
            queue.append((nxt, nodes + [nxt], new_edges))
    return found


def resolve(old_only, new_only, changed, prev, cur, lattice):
    adjacency = {}
    for eid, (h, donor, acceptor) in enumerate(changed):
        adjacency.setdefault(donor, []).append((acceptor, h, eid))
    candidates = {}
    for old in sorted(old_only):
        if lattice.get(old, False) or prev[old] != 1 or cur[old] != 2:
            continue
        for new in sorted(new_only):
            if lattice.get(new, False) or prev[new] != 2 or cur[new] != 1:
                continue
            paths = enumerate_paths(new, old, adjacency, prev, cur, lattice)
            if not paths:
                continue
            length = min(map(len, paths))
            best = [p for p in paths if len(p) == length]
            if len(best) == 1:
                candidates.setdefault(old, []).append((new, best[0]))
    remaining = {o: list(v) for o, v in candidates.items()}
    accepted = {}
    while True:
        users = defaultdict(list)
        for o, opts in remaining.items():
            for n, _ in opts:
                users[n].append(o)
        forced = [(o, opts[0]) for o, opts in remaining.items() if len(opts) == 1 and len(users[opts[0][0]]) == 1]
        if not forced:
            break
        used = {opt[0] for _, opt in forced}
        for o, opt in forced:
            accepted[o] = opt
            remaining.pop(o, None)
        for o in list(remaining):
            remaining[o] = [x for x in remaining[o] if x[0] not in used]
            if not remaining[o]:
                remaining.pop(o)
    conflicts = set()
    items = list(accepted.items())
    for i, (oi, (ni, pi)) in enumerate(items):
        ei = {e[2] for e in pi}; ii = {e[1] for e in pi[:-1]}
        for oj, (nj, pj) in items[i + 1:]:
            if ei & {e[2] for e in pj} or ii & {e[1] for e in pj[:-1]}:
                conflicts.update((oi, oj))
    for o in conflicts:
        remaining.setdefault(o, []).append(accepted.pop(o))
    return accepted, remaining


class Tracker:
    def __init__(self, first_frame, initial_carriers):
        self.first_frame = first_frame
        self.carriers = set(initial_carriers)
        self.lattice = {}
        self.tracks = []
        self.active = {}
        self.next_id = 0
        self.birth_counts = defaultdict(int); self.death_counts = defaultdict(int); self.event_counts = defaultdict(int)
        self.accepted_direct = self.accepted_relay = self.ambiguous_frames = 0
        self.checked = self.mismatches = 0
        for o in sorted(self.carriers):
            self.active[o] = self.new_track(o, first_frame, -1, "initial_frame", np.nan)
        self.birth_counts["initial_frame"] = len(self.active)

    def new_track(self, o, frame, reg, reason, z):
        t = dict(id=self.next_id, birth=frame, o0=o, o=o, reg0=reg, reg=reg, direct=0, relay=0, relay_h=0,
                 hop=np.zeros(3), path=0.0, path_xy=0.0, zmin=z, zmax=z, birth_reason=reason,
                 regmask=(1 << reg) if reg >= 0 else 0)
        self.next_id += 1
        return t

    def check(self, site_set):
        self.checked += 1
        self.mismatches += site_set != self.carriers

    def frame(self, fr, E):
        lattice = self.lattice
        prev = {}; cur = {}; zr = {}; row = {}
        changed = []; lost_o = set(); gained_o = set()
        for e in E:
            d, c = int(e["donor_id"]), int(e["acceptor_id"])
            if d >= 0:
                prev[d] = int(e["donor_nh0"]); cur[d] = int(e["donor_nh1"]); zr[d] = float(e["donor_zrel"]); row[d] = int(e["donor_row"])
            if c >= 0:
                prev[c] = int(e["acceptor_nh0"]); cur[c] = int(e["acceptor_nh1"]); zr[c] = float(e["acceptor_zrel"]); row[c] = int(e["acceptor_row"])
            if d >= 0 and c >= 0:
                changed.append((int(e["h_id"]), d, c))
            elif d >= 0:
                lost_o.add(d)
            elif c >= 0:
                gained_o.add(c)
        for o, r in row.items():
            lattice[o] = r != 0
        touched = set(prev)
        old_t = {o for o in touched if not lattice[o] and prev[o] == 1}
        new_t = {o for o in touched if not lattice[o] and cur[o] == 1}
        if old_t != (self.carriers & touched):
            raise RuntimeError(f"frame {fr}: event nH0 disagrees with the tracked carrier set")
        old_only, new_only = old_t - new_t, new_t - old_t
        accepted, unresolved = resolve(old_only, new_only, changed, prev, cur, lattice)
        unresolved_old = set(unresolved); unresolved_new = {n for opts in unresolved.values() for n, _ in opts}
        if unresolved:
            self.ambiguous_frames += 1
        hop_of = {(int(e["h_id"]), int(e["donor_id"]), int(e["acceptor_id"])): -e["hop"].astype(np.float64) for e in E}
        active = self.active
        nxt = {o: active.pop(o) for o in sorted(old_t & new_t)}            # stable carriers touched by this frame
        for o, t in nxt.items():
            t["regmask"] |= 1 << region(row[o], zr[o]); t["zmin"] = np.fmin(t["zmin"], zr[o]); t["zmax"] = np.fmax(t["zmax"], zr[o])
        edge_res = {}
        for old, (new, path) in accepted.items():
            t = active.pop(old)
            hop = np.zeros(3)
            for donor, acceptor, h in path:
                hop += hop_of[(h, donor, acceptor)]
                edge_res[h] = "canonical_direct" if len(path) == 1 else "canonical_relay"
            t["hop"] += hop; t["path"] += float(np.linalg.norm(hop)); t["path_xy"] += float(np.linalg.norm(hop[:2]))
            if len(path) == 1:
                t["direct"] += 1; self.accepted_direct += 1
            else:
                t["relay"] += 1; t["relay_h"] += len(path); self.accepted_relay += 1
            t["o"] = new; reg = region(row[new], zr[new]); t["reg"] = reg; t["regmask"] |= 1 << reg
            t["zmin"] = np.fmin(t["zmin"], zr[new]); t["zmax"] = np.fmax(t["zmax"], zr[new])
            nxt[new] = t
        for o in sorted(old_only):
            if o in accepted:
                continue
            t = active.pop(o)
            if o in unresolved_old:
                reason = "ambiguous_relay"
            elif any((dd == o and lattice[aa]) or (aa == o and lattice[dd]) for _, dd, aa in changed):
                reason = "surface_exchange"
            elif o in lost_o:
                reason = "assignment_gap"
            else:
                reason = {0: "OH_to_O2minus", 2: "OH_to_H2O", 3: "OH_to_H3Oplus"}.get(cur[o], "OH_identity_break")
            self.death_counts[reason] += 1
            t.update(death=fr - 1, death_reason=reason, reg_end=region(row[o], zr[o]))
            self.tracks.append(t)
        for o in sorted(new_t - set(nxt)):
            if o in unresolved_new:
                reason = "ambiguous_relay"
            elif any((dd == o and lattice[aa]) or (aa == o and lattice[dd]) for _, dd, aa in changed):
                reason = "surface_exchange"
            elif o in gained_o:
                reason = "assignment_gap"
            else:
                reason = {0: "O2minus_to_OH", 2: "H2O_to_OH", 3: "H3Oplus_to_OH"}.get(prev[o], "OH_identity_birth")
            self.birth_counts[reason] += 1
            nxt[o] = self.new_track(o, fr, region(row[o], zr[o]), reason, zr[o])
        active.update(nxt)
        self.carriers = (self.carriers - old_t) | new_t
        if set(active) != self.carriers:
            raise RuntimeError(f"frame {fr}: active/carrier mismatch")
        for h, d, c in changed:
            if h in edge_res:
                et = edge_res[h]
            elif lattice[d] or lattice[c]:
                et = "surface_exchange"
            elif 3 in (prev[d], cur[d], prev[c], cur[c]):
                et = "hydronium_involved"
            elif d in unresolved_new or c in unresolved_old:
                et = "ambiguous_candidate"
            else:
                et = "noncanonical_owner_change"
            self.event_counts[et] += 1
        for e in E:
            if e["acceptor_id"] < 0:
                self.event_counts["assignment_loss"] += 1
            elif e["donor_id"] < 0:
                self.event_counts["assignment_gain"] += 1

    def finish(self, last_frame):
        for o, t in sorted(self.active.items()):
            t.update(death=last_frame, death_reason="trajectory_end", reg_end=t["reg"])
            self.tracks.append(t)
        self.active = {}
        self.tracks.sort(key=lambda t: t["id"])


def site_sets_of(sites):
    """{frame: set of water OH- O ids} for every sampled frame of one segment"""
    frames = np.unique(sites["frame"])
    oh = sites[(sites["row"] == 0) & (sites["nh"] == 1)]
    oh = oh[np.argsort(oh["frame"], kind="stable")]
    b = np.r_[np.searchsorted(oh["frame"], frames), oh.size]
    ids = oh["id"]
    return {int(fr): set(ids[b[k]:b[k + 1]].tolist()) for k, fr in enumerate(frames)}


class ScheduleTracker(Tracker):
    """Tracker that also records every carrier-O assignment: (track id, first frame on that O, O id)."""

    def __init__(self, first_frame, initial_carriers):
        super().__init__(first_frame, initial_carriers)
        self.cur = {t["id"]: o for o, t in self.active.items()}
        self.changes = [(t["id"], first_frame, o) for o, t in sorted(self.active.items())]

    def frame(self, fr, E):
        super().frame(fr, E)
        for o, t in self.active.items():
            if self.cur.get(t["id"]) != o:
                self.cur[t["id"]] = o
                self.changes.append((t["id"], fr, o))


TRACK_DTYPE = np.dtype([("track_id", "<i4"), ("birth_frame", "<i8"), ("death_frame", "<i8"), ("initial_O", "<i4"),
                        ("final_O", "<i4"), ("initial_region", "i1"), ("final_region", "i1"), ("direct_hops", "<i4"),
                        ("relay_events", "<i4"), ("relay_H_transfers", "<i4"), ("hop_net_x", "<f8"), ("hop_net_y", "<f8"),
                        ("hop_net_z", "<f8"), ("hop_path_3d", "<f8"), ("hop_path_xy", "<f8"), ("zrel_min", "<f4"),
                        ("zrel_max", "<f4"), ("region_mask", "<i4")])
CHANGE_DTYPE = np.dtype([("track_id", "<i4"), ("frame", "<i8"), ("O", "<i4")])


@dataclass
class CarrierTrackResult:
    tracks: np.ndarray            # TRACK_DTYPE; alive frames birth_frame .. death_frame (inclusive, global indices)
    changes: np.ndarray           # CHANGE_DTYPE sorted by (track_id, frame): the carrier sits on O from that frame on
    birth_reason: np.ndarray
    death_reason: np.ndarray
    first_frame: int
    last_frame: int
    summary: dict
    extra: dict | None = None


def segment_frame_range(events_path: Path) -> tuple[int, int]:
    """(first, last) global frame covered by one site-current pass: its intervals start at first .. last - 1."""
    pass_npz = events_path.with_name(events_path.name.replace("_events.npz", ".npz"))
    if not pass_npz.is_file():
        raise FileNotFoundError(f"{pass_npz} (the site-current output next to {events_path.name}) is needed for the frame range")
    isf = np.load(pass_npz)["interval_start_frame"]
    return int(isf[0]), int(isf[-1]) + 1


def track_event_tables(files: Sequence[str | Path], log=None, positive: bool = False) -> CarrierTrackResult:
    """Run the strict tracker over the event tables of one run (segments in time order, contiguous).
    positive=True also tracks the H3O+ / Os_H family (result.extra: positive_tracks, pairs, positive_summary)."""
    tr = None
    last_frame = None
    for f in map(Path, files):
        with np.load(f) as z:
            if str(z["format"]) != "waterint-events-1":
                raise ValueError(f"{f}: not a waterint-events-1 table")
            ev = z["events"]
            ss = site_sets_of(z["sites"])
            if positive:
                sh, so_ = sets_of(z["sites"], 0, 3), sets_of(z["sites"], 1, 1)
        s0, s1 = segment_frame_range(f)
        if tr is None:
            if s0 not in ss:
                raise ValueError(f"{f}: no site record at the first frame {s0}")
            tr = ChargeTracker(s0, ss[s0], sh[s0], so_[s0]) if positive else ScheduleTracker(s0, ss[s0])
        elif s0 != last_frame:
            raise ValueError(f"{f}: not contiguous with the previous segment (starts at {s0}, expected {last_frame})")
        ev = ev[np.argsort(ev["frame"], kind="stable")]
        if ev.size and (int(ev["frame"][0]) <= s0 or int(ev["frame"][-1]) > s1):
            raise ValueError(f"{f}: event frames outside ({s0}, {s1}]")
        sframes = sorted(x for x in ss if x > s0)
        k = 0
        fstarts = np.flatnonzero(np.r_[True, ev["frame"][1:] != ev["frame"][:-1]]) if ev.size else np.zeros(0, int)
        for a, b in zip(fstarts, np.r_[fstarts[1:], ev.size]):
            fr = int(ev["frame"][a])
            while k < len(sframes) and sframes[k] < fr:
                tr.check(ss[sframes[k]])
                if positive:
                    tr.check_pos(sh[sframes[k]], so_[sframes[k]])
                k += 1
            tr.frame(fr, ev[a:b])
        while k < len(sframes):
            tr.check(ss[sframes[k]])
            if positive:
                tr.check_pos(sh[sframes[k]], so_[sframes[k]])
            k += 1
        last_frame = s1
        if log:
            log(f"{f}: frames {s0}-{s1}, tracks so far {tr.next_id}, site mismatches {tr.mismatches}/{tr.checked}")
    if tr is None:
        raise ValueError("no event tables given")
    if positive and tr.pmismatch:
        raise RuntimeError(f"tracked positive set disagrees with the site table at {tr.pmismatch} of {tr.pchecked} sampled frames")
    if tr.mismatches:
        raise RuntimeError(f"tracked carrier set disagrees with the site table at {tr.mismatches} of {tr.checked} sampled frames")
    tr.finish(last_frame)
    T = tr.tracks
    rec = np.array([(t["id"], t["birth"], t["death"], t["o0"], t["o"], t["reg0"], t["reg_end"], t["direct"], t["relay"],
                     t["relay_h"], *t["hop"], t["path"], t["path_xy"], t["zmin"], t["zmax"], t["regmask"]) for t in T],
                   dtype=TRACK_DTYPE)
    ch = np.array(tr.changes, dtype=CHANGE_DTYPE)
    ch = ch[np.lexsort((ch["frame"], ch["track_id"]))]
    summary = dict(first_frame=tr.first_frame, last_frame=last_frame, frames=last_frame - tr.first_frame + 1,
                   tracks=tr.next_id, accepted_direct_hops=tr.accepted_direct, accepted_relay_events=tr.accepted_relay,
                   ambiguous_frames=tr.ambiguous_frames,
                   event_counts=dict(sorted((k, v) for k, v in tr.event_counts.items() if v)),
                   birth_counts=dict(sorted(tr.birth_counts.items())), death_counts=dict(sorted(tr.death_counts.items())),
                   site_checks=tr.checked, site_mismatches=int(tr.mismatches), schedule_rows=int(ch.size))
    res = CarrierTrackResult(rec, ch, np.array([t["birth_reason"] for t in T]), np.array([t["death_reason"] for t in T]),
                             tr.first_frame, last_frame, summary)
    if positive:
        P = tr.ptracks
        res.extra = {
            "positive_tracks": np.array([(t["id"], t["birth"], t["death"], t["o0"], t["o"], t["kind0"], t["kind"], t["direct"], t["relay"],
                                          t["conversions"], *t["hop"], t["birth_reason"], t["death_reason"]) for t in P], dtype=POSITIVE_DTYPE),
            "pairs": np.array(tr.pairs, dtype=PAIR_DTYPE),
            "positive_summary": dict(positive_tracks=tr.pnext, positive_counts=dict(sorted(tr.pcounts.items())), pairs=len(tr.pairs),
                                     pair_creation=int(sum(p[1] == 0 for p in tr.pairs)), recombination=int(sum(p[1] == 1 for p in tr.pairs)),
                                     positive_site_mismatches=int(tr.pmismatch), positive_site_checks=int(tr.pchecked))}
    return res


# ---------------------------------------------------------------- positive family (H3O+ / Os_H) tracker
# One family of +1 sites relative to the reference (water nH 2, solid O nH 0): H3O+ = water O with nH 3, Os_H = solid O
# with nH 1. Moves mirror the OH- rule with the proton direction reversed (old positive site -> new positive site by the
# unique shortest path of <= 4 changed O-H edges, internal O water nH 2 in both frames, forced one-to-one matching);
# a positive track continues through H3O+ <-> Os_H conversions. Births / deaths: pair_creation_with_OH,
# recombination_with_OH, ambiguous_relay, assignment_gap, identity_break, other. Checked against the site table.
POS_OLD = {(0, 3, 2), (1, 1, 0)}      # (row, prev nH, cur nH) of a positive site that loses its excess proton
POS_NEW = {(0, 2, 3), (1, 0, 1)}


def kind(row, nh):
    if row == 0 and nh == 3:
        return "H3O+"
    if row != 0 and nh == 1:
        return "Os_H"
    return None


def resolve_pos(old_only, new_only, changed, prev, cur, row, lattice):
    adjacency = {}
    for eid, (h, donor, acceptor) in enumerate(changed):
        adjacency.setdefault(donor, []).append((acceptor, h, eid))
    candidates = {}
    for old in sorted(old_only):
        if (row[old], prev[old], cur[old]) not in POS_OLD:
            continue
        for new in sorted(new_only):
            if (row[new], prev[new], cur[new]) not in POS_NEW:
                continue
            paths = enumerate_paths(old, new, adjacency, prev, cur, lattice)
            if not paths:
                continue
            length = min(map(len, paths))
            best = [p for p in paths if len(p) == length]
            if len(best) == 1:
                candidates.setdefault(old, []).append((new, best[0]))
    remaining = {o: list(v) for o, v in candidates.items()}
    accepted = {}
    while True:
        users = defaultdict(list)
        for o, opts in remaining.items():
            for n, _ in opts:
                users[n].append(o)
        forced = [(o, opts[0]) for o, opts in remaining.items() if len(opts) == 1 and len(users[opts[0][0]]) == 1]
        if not forced:
            break
        used = {opt[0] for _, opt in forced}
        for o, opt in forced:
            accepted[o] = opt
            remaining.pop(o, None)
        for o in list(remaining):
            remaining[o] = [x for x in remaining[o] if x[0] not in used]
            if not remaining[o]:
                remaining.pop(o)
    conflicts = set()
    items = list(accepted.items())
    for i, (oi, (ni, pi)) in enumerate(items):
        ei = {e[2] for e in pi}; ii = {e[1] for e in pi[:-1]}
        for oj, (nj, pj) in items[i + 1:]:
            if ei & {e[2] for e in pj} or ii & {e[1] for e in pj[:-1]}:
                conflicts.update((oi, oj))
    for o in conflicts:
        remaining.setdefault(o, []).append(accepted.pop(o))
    return accepted, remaining


class ChargeTracker(ScheduleTracker):
    """OH- tracking (with the carrier-O schedule) unchanged, plus the positive family."""

    def __init__(self, first_frame, oh, h3o, osh):
        super().__init__(first_frame, oh)
        self.pos = {o: "H3O+" for o in h3o} | {o: "Os_H" for o in osh}     # site -> kind
        self.pactive = {}
        self.ptracks = []
        self.pnext = 0
        self.pairs = []                     # (frame, type, positive site, OH- site, positive kind)
        self.pcounts = defaultdict(int)
        self.pchecked = self.pmismatch = 0
        for o in sorted(self.pos):
            self.pactive[o] = self.new_ptrack(o, first_frame, self.pos[o], "initial_frame")
        self.pcounts["birth:initial_frame"] = len(self.pactive)

    def new_ptrack(self, o, frame, k, reason):
        t = dict(id=self.pnext, birth=frame, o0=o, o=o, kind0=k, kind=k, direct=0, relay=0, conversions=0,
                 hop=np.zeros(3), birth_reason=reason)
        self.pnext += 1
        return t

    def check_pos(self, h3o_set, osh_set):
        self.pchecked += 1
        mine_h = {o for o, k in self.pos.items() if k == "H3O+"}; mine_s = {o for o, k in self.pos.items() if k == "Os_H"}
        self.pmismatch += (mine_h != h3o_set) or (mine_s != osh_set)

    def frame(self, fr, E):
        oh_before = set(self.carriers)
        super().frame(fr, E)                                   # OH- rules, updates self.carriers/lattice
        oh_died = oh_before - self.carriers
        oh_born = self.carriers - oh_before
        prev = {}; cur = {}; row = {}
        changed = []; lost = set(); gained = set()
        for e in E:
            d, c = int(e["donor_id"]), int(e["acceptor_id"])
            if d >= 0:
                prev[d] = int(e["donor_nh0"]); cur[d] = int(e["donor_nh1"]); row[d] = int(e["donor_row"])
            if c >= 0:
                prev[c] = int(e["acceptor_nh0"]); cur[c] = int(e["acceptor_nh1"]); row[c] = int(e["acceptor_row"])
            if d >= 0 and c >= 0:
                changed.append((int(e["h_id"]), d, c))
            elif d >= 0:
                lost.add(d)
            elif c >= 0:
                gained.add(c)
        touched = set(prev)
        old_t = {o for o in touched if kind(row[o], prev[o])}
        new_t = {o for o in touched if kind(row[o], cur[o])}
        if old_t != (set(self.pos) & touched):
            raise RuntimeError(f"frame {fr}: event nH0 disagrees with the tracked positive set")
        stable = {o for o in old_t & new_t if kind(row[o], prev[o]) == kind(row[o], cur[o])}
        old_only, new_only = old_t - stable, new_t - stable
        accepted, unresolved = resolve_pos(old_only, new_only, changed, prev, cur, row, self.lattice)
        unres_old = set(unresolved); unres_new = {n for opts in unresolved.values() for n, _ in opts}
        hop_of = {(int(e["h_id"]), int(e["donor_id"]), int(e["acceptor_id"])): e["hop"].astype(np.float64) for e in E}
        act = self.pactive
        nxt = {o: act.pop(o) for o in stable}
        for old, (new, path) in accepted.items():
            t = act.pop(old)
            for donor, acceptor, h in path:
                t["hop"] += hop_of[(h, donor, acceptor)]          # positive charge moves with the proton
            t["direct" if len(path) == 1 else "relay"] += 1
            k = kind(row[new], cur[new]); t["conversions"] += k != t["kind"]; t["kind"] = k; t["o"] = new
            nxt[new] = t
            self.pcounts["move_direct" if len(path) == 1 else "move_relay"] += 1
            if k != kind(row[old], prev[old]):
                self.pcounts[f"convert:{kind(row[old], prev[old])}->{k}"] += 1
        edges_from = defaultdict(set); edges_to = defaultdict(set)
        for h, d, c in changed:
            edges_from[d].add(c); edges_to[c].add(d)
        for o in sorted(old_only - set(accepted)):
            t = act.pop(o)
            partner = sorted(edges_from[o] & oh_died)
            if o in unres_old:
                reason = "ambiguous_relay"
            elif partner:
                reason = "recombination_with_OH"; self.pairs.append((fr, 1, o, partner[0], t["kind"]))
            elif o in lost:
                reason = "assignment_gap"
            elif kind(row[o], cur[o]) is None and (row[o], prev[o], cur[o]) not in POS_OLD:
                reason = "identity_break"
            else:
                reason = "other"
            t.update(death=fr - 1, death_reason=reason); self.ptracks.append(t)
            self.pcounts[f"death:{reason}"] += 1
        for o in sorted(new_t - set(nxt)):
            k = kind(row[o], cur[o])
            partner = sorted(edges_to[o] & oh_born)
            if o in unres_new:
                reason = "ambiguous_relay"
            elif partner:
                reason = "pair_creation_with_OH"; self.pairs.append((fr, 0, o, partner[0], k))
            elif o in gained:
                reason = "assignment_gap"
            elif (row[o], prev[o], cur[o]) not in POS_NEW:
                reason = "identity_break"
            else:
                reason = "other"
            nxt[o] = self.new_ptrack(o, fr, k, reason)
            self.pcounts[f"birth:{reason}"] += 1
        act.update(nxt)
        for o in touched:
            k = kind(row[o], cur[o])
            if k:
                self.pos[o] = k
            else:
                self.pos.pop(o, None)
        if set(act) != set(self.pos):
            raise RuntimeError(f"frame {fr}: positive active/site mismatch")

    def finish(self, last_frame):
        super().finish(last_frame)
        for o, t in sorted(self.pactive.items()):
            t.update(death=last_frame, death_reason="trajectory_end"); self.ptracks.append(t)
        self.pactive = {}
        self.ptracks.sort(key=lambda t: t["id"])


def sets_of(sites, rowv, nhv):
    frames = np.unique(sites["frame"])
    s = sites[(sites["row"] == rowv) & (sites["nh"] == nhv)]
    s = s[np.argsort(s["frame"], kind="stable")]
    b = np.r_[np.searchsorted(s["frame"], frames), s.size]
    return {int(fr): set(s["id"][b[k]:b[k + 1]].tolist()) for k, fr in enumerate(frames)}


POSITIVE_DTYPE = np.dtype([("track_id", "<i4"), ("birth_frame", "<i8"), ("death_frame", "<i8"), ("initial_O", "<i4"),
                           ("final_O", "<i4"), ("initial_kind", "U4"), ("final_kind", "U4"), ("direct_moves", "<i4"),
                           ("relay_moves", "<i4"), ("conversions", "<i4"), ("hop_net_x", "<f8"), ("hop_net_y", "<f8"),
                           ("hop_net_z", "<f8"), ("birth_reason", "U24"), ("death_reason", "U24")])
PAIR_DTYPE = np.dtype([("frame", "<i8"), ("type", "i1"), ("positive_O", "<i4"), ("OH_O", "<i4"), ("positive_kind", "U4")])


def schedule_starts(changes: np.ndarray, n_tracks: int) -> np.ndarray:
    return np.searchsorted(changes["track_id"], np.arange(n_tracks + 1))


def carrier_O_at(changes: np.ndarray, starts: np.ndarray, track: int, frames: np.ndarray) -> np.ndarray:
    """O id of `track` at the given frames (inside its life); `starts` = schedule_starts(changes, n_tracks)."""
    lo, hi = starts[track], starts[track + 1]
    return changes["O"][lo:hi][np.searchsorted(changes["frame"][lo:hi], frames, side="right") - 1]
