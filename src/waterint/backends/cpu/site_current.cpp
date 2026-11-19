// WaterInt: complete formal-ion site current (streaming, chunked input).
//
// Per-frame arithmetic: H assignment, unwrapping, dM_ref, B, dM_site, the
// motion/transfer/image decomposition, every balance and closure check.
// Buffers are persistent (grid/assignment buffers, swaps instead of copies): frames are pushed in blocks from Python
// (LAMMPS text through the reader below, or decoded GPUMD quantized npz), and
// records are written into caller-owned arrays.
//
// General definition (the system profile supplies the numbers, e.g. mgo_water):
// every atom has a formal charge q_i (MgO: H +1, O -2, Mg +2). Atoms have a role: proton (H), acceptor
// (atoms that can own H, e.g. O; site charge = q_acceptor + nH) or fixed-charge ion (e.g. Mg, Na, Cl).
// Each H is owned by the nearest acceptor within `cutoff` (MgO 1.50 A) under the minimum image of the
// periodic axes, ties within `tie_tol` (MgO 1e-8 A) -> smallest atom ID; free H keep +1.
//   dM_ref/e  = sum_i q_i * MIC(dr_i)
//   B/e       = sum_{assigned H} MIC(r_H - r_owner)
//   dM_site   = dM_ref - dB = motion + transfer + image   (checked every interval)
// Optional decomposition: dM_site split into (group, z-window) cells. Site motion of an acceptor is split
// half/half between its classes at the two ends of the interval; windows use the interval-midpoint height
// relative to a per-frame reference plane; H transfers enter as minimum-image hop vectors (whole-cell
// unwrap remainders go to the image group); the cells sum to dM_site (checked).
// Orthorhombic cell; x/y periodic; z periodic or open.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct V3 { double x = 0, y = 0, z = 0; };
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
V3& operator+=(V3& a, V3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
V3& operator-=(V3& a, V3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
double nrm(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
double maxabs(V3 a) { return std::max({std::abs(a.x), std::abs(a.y), std::abs(a.z)}); }
V3 mic(V3 a, double lx, double ly) { a.x -= lx * std::nearbyint(a.x / lx); a.y -= ly * std::nearbyint(a.y / ly); return a; }
void need(bool b, const std::string& m) { if (!b) throw std::runtime_error(m); }

thread_local std::string g_error;

#pragma pack(push, 1)
struct FrameRec { int64_t frame, step; double time_ps, B[3]; int32_t assigned, free, ties, formal_q; double max_assigned_dist; };
struct IntervalRec {
    int64_t frame_start, step_start; double time_mid;
    double dMref[3], dB[3], dMsite[3], motion[3], transfer[3], dI[3], residual[3];
    int32_t assigned0, free0, ties0, assigned1, free1, ties1, owner_changes, crossings[3];
    double max_mic_component, max_mic_norm, max_raw_vel, max_raw_vxy_dt_over_L;
};
#pragma pack(pop)
// Event table records (format waterint-events-1).
// One EventRec per H owner change (owner -1 = free H): nH and species-table rows of both acceptors before (0) and
// after (1) the interval, interval-midpoint z_rel of both sites, minimum-image hop vector between the two sites
// (interval midpoints; zero when one side is a free H) and the H distance to its new owner (-1 if free).
#pragma pack(push, 1)
struct EventRec {
    int64_t frame; int32_t h_id, donor_id, acceptor_id; int8_t donor_row, acceptor_row;
    int16_t donor_nh0, donor_nh1, acceptor_nh0, acceptor_nh1; float donor_zrel, acceptor_zrel, hop[3], d_new;
};
// Tracked acceptor site (e.g. OH-, H3O+, Os-H) at a sampled frame: unwrapped position (A).
struct SiteRec { int64_t frame; int32_t id; int8_t row; int8_t nh; float u[3]; };
#pragma pack(pop)
static_assert(sizeof(EventRec) == 54, "EventRec layout");
static_assert(sizeof(SiteRec) == 26, "SiteRec layout");
static_assert(sizeof(FrameRec) == 72, "FrameRec layout");
static_assert(sizeof(IntervalRec) == 264, "IntervalRec layout");

struct Assign { std::vector<int> owner, nh; int ties = 0, assigned = 0; double maxdist = 0; };

struct Grid {
    int nx, ny, nz; double xlo, ylo, zlo, lx, ly, lz, wx, wy, wz; std::vector<int> head, next; bool pz = false;
    Grid(double x, double y, double z, double Lx, double Ly, double Lz, int no, double cell = 1.5 + 1e-8, bool periodic_z = false)
        : xlo(x), ylo(y), zlo(z), lx(Lx), ly(Ly), lz(Lz), pz(periodic_z) {
        nx = std::max(1, int(std::floor(lx / cell))); ny = std::max(1, int(std::floor(ly / cell))); nz = std::max(1, int(std::floor(lz / cell)));
        wx = lx / nx; wy = ly / ny; wz = lz / nz; head.resize(size_t(nx) * ny * nz); next.resize(no);
    }
    int cell(double v, double lo, double L, double w, int n, bool per) const {
        (void)L; int k = int(std::floor((v - lo) / w)); if (per) { k %= n; if (k < 0) k += n; } else k = std::clamp(k, 0, n - 1); return k;
    }
    void build(const std::vector<V3>& r, const std::vector<int>& os) {
        std::fill(head.begin(), head.end(), -1);
        for (size_t j = 0; j < os.size(); ++j) {
            V3 p = r[os[j]];
            int x = cell(p.x, xlo, lx, wx, nx, true), y = cell(p.y, ylo, ly, wy, ny, true), z = cell(p.z, zlo, lz, wz, nz, pz), c = (z * ny + y) * nx + x;
            next[j] = head[c]; head[c] = int(j);
        }
    }
    template <class F> void visit(V3 h, const std::vector<int>& os, F fn) const {
        int ix = cell(h.x, xlo, lx, wx, nx, true), iy = cell(h.y, ylo, ly, wy, ny, true), iz = cell(h.z, zlo, lz, wz, nz, pz), xs[3], ys[3], xn = 0, yn = 0;
        for (int d = -1; d <= 1; ++d) {
            int x = (ix + d) % nx; if (x < 0) x += nx; bool seen = false; for (int k = 0; k < xn; ++k) if (xs[k] == x) seen = true; if (!seen) xs[xn++] = x;
            int y = (iy + d) % ny; if (y < 0) y += ny; seen = false; for (int k = 0; k < yn; ++k) if (ys[k] == y) seen = true; if (!seen) ys[yn++] = y;
        }
        int zs[3], zn = 0;
        for (int d = -1; d <= 1; ++d) {
            int z = iz + d;
            if (pz) { z %= nz; if (z < 0) z += nz; } else if (z < 0 || z >= nz) continue;
            bool seen = false; for (int k = 0; k < zn; ++k) if (zs[k] == z) seen = true; if (!seen) zs[zn++] = z;
        }
        for (int jz = 0; jz < zn; ++jz) {
            int z = zs[jz];
            for (int jy = 0; jy < yn; ++jy) for (int jx = 0; jx < xn; ++jx) {
                int c = (z * ny + ys[jy]) * nx + xs[jx];
                for (int k = head[c]; k != -1; k = next[k]) fn(os[k]);
            }
        }
    }
};

struct Stats {  // order is the Python-side STATS_FIELDS
    double max_box_drift = 0, max_step_comp = 0, max_step_norm = 0, max_vel = 0, max_vel_ratio = 0;
    double crossed_x = 0, crossed_y = 0, frames = 0, first_step = 0, last_step = 0, formal_q = 0, owner_changes = 0, ties = 0, free_h_samples = 0;
    double nl_list_frames = 0, nl_full_frames = 0, nl_rebuilds = 0, nl_audits = 0, nl_fallbacks = 0;
};

class SiteCurrent {
public:
    // role: 1 = proton (H), 2 = acceptor (may own H), 0 = fixed-charge atom. qf: formal charge of every atom.
    SiteCurrent(int nat, const int32_t* ids_in, const int8_t* role_in, const double* qf_in, const double* box6, const uint8_t* pbc3,
                double cutoff, double tie_tol, int64_t first_step, int64_t dstep, double dt_ps, double t0_ps)
        : nat_(nat), dstep_(dstep), dt_(dt_ps), t0_(t0_ps), firststep_(first_step) {
        need(nat > 0, "natoms must be positive"); need(dstep > 0 && dt_ps > 0, "dstep and dt must be positive");
        need(cutoff > 0 && tie_tol >= 0 && tie_tol < cutoff, "invalid H cutoff / tie tolerance");
        need(pbc3[0] && pbc3[1], "x and y must be periodic"); pz_ = pbc3[2] != 0;
        cutoff_ = cutoff; tol_ = tie_tol; cut_hi_ = cutoff + tie_tol;
        ids_.assign(ids_in, ids_in + nat); role_.assign(role_in, role_in + nat); qf_.assign(qf_in, qf_in + nat);
        for (int k = 0; k < 6; ++k) box_[k] = box6[k];
        for (int k = 0; k < 3; ++k) need(box_[2 * k + 1] - box_[2 * k] > 3.0, "box too small");
        double qsum = 0, qbase = 0;
        for (int i = 0; i < nat; ++i) {
            need(role_[i] >= 0 && role_[i] <= 2, "atom role must be 0 (fixed), 1 (proton) or 2 (acceptor)");
            need(std::isfinite(qf_[i]) && qf_[i] == std::nearbyint(qf_[i]), "formal charges must be integers");
            if (role_[i] == 1) { need(qf_[i] == 1.0, "protons must have formal charge +1"); H_.push_back(i); }
            else if (role_[i] == 0) { Mg_.push_back(i); qbase += qf_[i]; }
            else { O_.push_back(i); qacc_.push_back(qf_[i]); qbase += qf_[i]; }
            qsum += qf_[i];
        }
        need(!H_.empty() && !O_.empty(), "need at least one proton and one acceptor");
        oordinal_.assign(nat, -1); for (size_t j = 0; j < O_.size(); ++j) oordinal_[O_[j]] = int(j);
        qformal_ = int64_t(std::llround(qsum)); qsite_base_ = int64_t(std::llround(qbase));
        stats_.first_step = double(first_step); stats_.formal_q = double(qformal_);
        grid_ = Grid(box_[0], box_[2], box_[4], box_[1] - box_[0], box_[3] - box_[2], box_[5] - box_[4], int(O_.size()), cut_hi_, pz_);
        cand_.reserve(32); flow_.assign(O_.size(), 0); bondNow_.assign(H_.size(), V3{});
        curRaw_.resize(nat_); curV_.assign(nat_, V3{}); curU_.resize(nat_); prevRaw_.resize(nat_); prevU_.resize(nat_);
    }

    // Decomposition setup. ref_mode: 0 none (z_ref = 0), 1 top layer, 2 bottom layer, 3 fixed value, 4 mean.
    // An acceptor atom i uses table row acc_row[i] (its element, and solid or not); its group with nH protons is
    // table[acc_row[i] * (nh_max + 1) + min(nH, nh_max)]. other_group[i] = group of a fixed-charge atom.
    // Windows are half-open [lo, hi) in z - z_ref; heights in no window go to slot n_windows.
    void set_decomposition(const uint8_t* ref_mask, int ref_mode, double ref_width, double ref_fixed, const int32_t* acc_row,
                           int n_rows, int nh_max, const int32_t* table, const int32_t* other_group, int free_h_group, int transfer_group,
                           int image_group, int n_groups, int n_windows, const double* edges) {
        need(n_groups > 0 && n_windows >= 0 && nh_max >= 0, "invalid decomposition sizes");
        ref_mode_ = ref_mode; ref_width_ = ref_width; ref_fixed_ = ref_fixed; ref_idx_.clear();
        for (int i = 0; i < nat_; ++i) if (ref_mask && ref_mask[i]) ref_idx_.push_back(i);
        need(ref_mode == 0 || ref_mode == 3 || !ref_idx_.empty(), "reference plane needs at least one atom");
        need(n_rows > 0, "need at least one species table row");
        nh_max_ = nh_max; acc_table_.assign(table, table + size_t(n_rows) * (nh_max + 1));
        acc_row_.assign(acc_row, acc_row + nat_);
        for (int i : O_) need(acc_row_[i] >= 0 && acc_row_[i] < n_rows, "acceptor without a valid species table row");
        other_group_.assign(other_group, other_group + nat_);
        G_ = n_groups; W_ = n_windows; edges_.assign(edges, edges + 2 * n_windows);
        auto ok = [&](int g) { return g >= 0 && g < G_; };
        for (int g : acc_table_) need(ok(g), "acceptor group out of range");
        for (int i : Mg_) need(ok(other_group_[i]), "fixed-charge atom without a valid group");
        need(ok(free_h_group) && ok(transfer_group) && ok(image_group), "special group out of range");
        g_free_ = free_h_group; g_transfer_ = transfer_group; g_image_ = image_group;
        dec_ = true; dec_buf_.assign(size_t(G_) * (W_ + 1) * 3, 0.0);
    }
    int decomposition_cells() const { return dec_ ? G_ * (W_ + 1) : 0; }

    // Event recording; needs the decomposition (groups, windows, species rows). counts/sites every `stride` frames.
    void set_events(bool enable, int counts_stride, int sites_stride, const uint8_t* tracked) {
        need(!enable || dec_, "events need the decomposition (profile species rows, groups and windows)");
        need(counts_stride > 0 && sites_stride > 0, "event strides must be positive");
        ev_ = enable; counts_stride_ = counts_stride; sites_stride_ = sites_stride;
        tracked_.assign(tracked, tracked + acc_table_.size());
    }
    std::vector<EventRec> events_; std::vector<int64_t> count_frames_; std::vector<int32_t> counts_; std::vector<SiteRec> sites_;

    // Push one frame (positions/velocities in internal atom order). Writes one FrameRec and,
    // for every frame after the first, one IntervalRec. Returns true if an interval was written.
    bool push(const double* pos, const double* vel, int64_t step, const double* bb, FrameRec& fr, IntervalRec& ir, double* dec_out, double* zref_out) {
        curRaw_.resize(nat_);
        const bool prev_had_vel = had_vel_; had_vel_ = vel != nullptr;
        for (int i = 0; i < nat_; ++i) {
            curRaw_[i] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
            need(std::isfinite(curRaw_[i].x) && std::isfinite(curRaw_[i].y) && std::isfinite(curRaw_[i].z), "nonfinite position");
            if (vel) curV_[i] = {vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]}; else if (prev_had_vel) curV_[i] = V3{};
        }
        const int64_t f = f_;
        bool wrote = false;
        if (dec_) { zref_cur_ = zref_now(); if (zref_out) *zref_out = zref_cur_; }
        if (f == 0) {
            need(step == firststep_, "first step differs from the declared first step");
            if (bb) for (int k = 0; k < 6; ++k) { stats_.max_box_drift = std::max(stats_.max_box_drift, std::abs(bb[k] - box_[k])); }
            need(stats_.max_box_drift <= 1e-9, "box drift exceeds 1e-9 A");
            curU_ = curRaw_;
            Assign& a = assign(grid_);
            int nhsum = 0; for (int j = 0; j < int(a.nh.size()); ++j) nhsum += a.nh[j];
            need(nhsum == a.assigned, "first-frame assigned-H/node count mismatch");
            int64_t qsite = int64_t(nhsum) + qsite_base_ + (int64_t(H_.size()) - a.assigned);
            need(qsite == qformal_, "first-frame formal site charge mismatch");
            for (int oi : a.owner) need(oi < 0 || oi < int(O_.size()), "invalid first-frame owner ordinal");
            for (V3 v : curV_) { stats_.max_vel = std::max(stats_.max_vel, maxabs(v)); stats_.max_vel_ratio = std::max(stats_.max_vel_ratio, std::max(std::abs(v.x) * dt_ / (box_[1] - box_[0]), std::abs(v.y) * dt_ / (box_[3] - box_[2]))); }
            prevOwner_ = a.owner; prevNH_ = a.nh; prevAssigned_ = a.assigned; prevTies_ = a.ties;
            prevBond_.assign(H_.size(), V3{});
            for (size_t h = 0; h < H_.size(); ++h) if (prevOwner_[h] >= 0) prevBond_[h] = micz(curRaw_[H_[h]] - curRaw_[O_[prevOwner_[h]]]);
        } else {
            need(step == firststep_ + dstep_ * f, "step increment is not exactly dstep");
            if (bb) for (int k = 0; k < 6; ++k) stats_.max_box_drift = std::max(stats_.max_box_drift, std::abs(bb[k] - box_[k]));
            need(stats_.max_box_drift <= 1e-9, "box drift exceeds 1e-9 A");
            double lx = box_[1] - box_[0], ly = box_[3] - box_[2];
            curU_.resize(nat_); V3 dMref{}; ir = IntervalRec{}; ir.frame_start = f - 1; ir.step_start = step - dstep_; ir.time_mid = t0_ + (f - 0.5) * dt_;
            int cros[3] = {0, 0, 0}; double micComp = 0, micNorm = 0, vmax = 0, vratio = 0;
            for (int i = 0; i < nat_; ++i) {
                V3 rawd = curRaw_[i] - prevRaw_[i], d = micz(rawd); curU_[i] = prevU_[i] + d;
                micComp = std::max(micComp, maxabs(d)); micNorm = std::max(micNorm, nrm(d));
                double sx = std::nearbyint((rawd.x - d.x) / lx), sy = std::nearbyint((rawd.y - d.y) / ly);
                cros[0] += int(std::abs(sx)); cros[1] += int(std::abs(sy));
                vmax = std::max(vmax, maxabs(curV_[i])); vratio = std::max(vratio, std::max(std::abs(curV_[i].x) * dt_ / lx, std::abs(curV_[i].y) * dt_ / ly));
                double q = qf_[i]; dMref += d * q;
            }
            stats_.crossed_x += cros[0]; stats_.crossed_y += cros[1];
            stats_.max_step_comp = std::max(stats_.max_step_comp, micComp); stats_.max_step_norm = std::max(stats_.max_step_norm, micNorm);
            stats_.max_vel = std::max(stats_.max_vel, vmax); stats_.max_vel_ratio = std::max(stats_.max_vel_ratio, vratio);
            Assign& a = assign(grid_);
            int nhsum = 0; for (int j = 0; j < int(a.nh.size()); ++j) nhsum += a.nh[j];
            need(nhsum == a.assigned, "assigned-H/node count mismatch");
            int64_t qsite = int64_t(nhsum) + qsite_base_ + (int64_t(H_.size()) - a.assigned);
            need(qsite == qformal_, "formal site charge changed");
            for (int oi : a.owner) need(oi < 0 || oi < int(O_.size()), "invalid owner ordinal");
            std::vector<int>& flow = flow_; std::fill(flow.begin(), flow.end(), 0); int freeFlow = 0;
            for (size_t h = 0; h < H_.size(); ++h) if (prevOwner_[h] != a.owner[h]) {
                if (prevOwner_[h] >= 0) --flow[prevOwner_[h]]; if (a.owner[h] >= 0) ++flow[a.owner[h]];
                if (prevOwner_[h] >= 0 && a.owner[h] < 0) ++freeFlow; if (prevOwner_[h] < 0 && a.owner[h] >= 0) --freeFlow;
            }
            for (size_t j = 0; j < O_.size(); ++j) need(a.nh[j] - prevNH_[j] == flow[j], "per-O nodewise H-owner balance failed");
            need((int(H_.size()) - a.assigned) - (int(H_.size()) - prevAssigned_) == freeFlow, "free-H nodewise owner balance failed");
            V3 Bnow{}; std::vector<V3>& bondNow = bondNow_; std::fill(bondNow.begin(), bondNow.end(), V3{});
            for (size_t h = 0; h < H_.size(); ++h) if (a.owner[h] >= 0) { bondNow[h] = micz(curRaw_[H_[h]] - curRaw_[O_[a.owner[h]]]); Bnow += bondNow[h]; }
            V3 Bprev{}; for (V3 b : prevBond_) Bprev += b;
            V3 dB = Bnow - Bprev, dMsite = dMref - dB, motion{}, transfer{}, dI{};
            for (size_t j = 0; j < O_.size(); ++j) { double q0 = qacc_[j] + prevNH_[j], q1 = qacc_[j] + a.nh[j]; motion += (curU_[O_[j]] - prevU_[O_[j]]) * (0.5 * (q0 + q1)); }
            for (int i : Mg_) motion += (curU_[i] - prevU_[i]) * qf_[i];
            int och = 0;
            for (size_t h = 0; h < H_.size(); ++h) {
                int o0 = prevOwner_[h], o1 = a.owner[h], hi = H_[h]; double free0 = o0 < 0 ? 1.0 : 0.0, free1 = o1 < 0 ? 1.0 : 0.0;
                if (free0 || free1) motion += (curU_[hi] - prevU_[hi]) * (0.5 * (free0 + free1));
                int node0 = o0 < 0 ? hi : O_[o0], node1 = o1 < 0 ? hi : O_[o1];
                if (o0 != o1) { ++och; transfer += ((curU_[node1] + prevU_[node1]) - (curU_[node0] + prevU_[node0])) * 0.5; }
                if (o0 >= 0) dI -= (prevU_[hi] - prevBond_[h] - prevU_[O_[o0]]);
                if (o1 >= 0) dI += (curU_[hi] - bondNow[h] - curU_[O_[o1]]);
            }
            V3 res = dMsite - motion - transfer - dI; need(maxabs(res) <= 1e-8, "independent site motion/edge/image closure failed");
            if (dec_) decompose(a, dMsite, dec_out);
            if (ev_ && och > 0) record_events(a, f);
            ir.dMref[0] = dMref.x; ir.dMref[1] = dMref.y; ir.dMref[2] = dMref.z; ir.dB[0] = dB.x; ir.dB[1] = dB.y; ir.dB[2] = dB.z;
            ir.dMsite[0] = dMsite.x; ir.dMsite[1] = dMsite.y; ir.dMsite[2] = dMsite.z; ir.motion[0] = motion.x; ir.motion[1] = motion.y; ir.motion[2] = motion.z;
            ir.transfer[0] = transfer.x; ir.transfer[1] = transfer.y; ir.transfer[2] = transfer.z; ir.dI[0] = dI.x; ir.dI[1] = dI.y; ir.dI[2] = dI.z;
            ir.residual[0] = res.x; ir.residual[1] = res.y; ir.residual[2] = res.z;
            ir.assigned0 = prevAssigned_; ir.free0 = int(H_.size()) - prevAssigned_; ir.ties0 = prevTies_; ir.assigned1 = a.assigned; ir.free1 = int(H_.size()) - a.assigned; ir.ties1 = a.ties;
            ir.owner_changes = och; ir.crossings[0] = cros[0]; ir.crossings[1] = cros[1]; ir.crossings[2] = 0;
            ir.max_mic_component = micComp; ir.max_mic_norm = micNorm; ir.max_raw_vel = vmax; ir.max_raw_vxy_dt_over_L = vratio;
            stats_.owner_changes += och;
            prevOwner_.swap(a.owner); prevNH_.swap(a.nh); prevAssigned_ = a.assigned; prevTies_ = a.ties; prevBond_.swap(bondNow);
            wrote = true;
        }
        V3 B{}; for (V3 b : prevBond_) B += b;
        fr = FrameRec{}; fr.frame = f; fr.step = step; fr.time_ps = t0_ + f * dt_; fr.B[0] = B.x; fr.B[1] = B.y; fr.B[2] = B.z;
        fr.assigned = prevAssigned_; fr.free = int(H_.size()) - prevAssigned_; fr.ties = prevTies_; fr.formal_q = int32_t(qformal_); fr.max_assigned_dist = 0;
        for (size_t h = 0; h < H_.size(); ++h) if (prevOwner_[h] >= 0) { double d = nrm(micz(curRaw_[H_[h]] - curRaw_[O_[prevOwner_[h]]])); fr.max_assigned_dist = std::max(fr.max_assigned_dist, d); }
        stats_.ties += prevTies_; stats_.free_h_samples += fr.free;
        if (ev_) record_frame(f);
        zref_prev_ = zref_cur_;
        prevRaw_.swap(curRaw_); prevU_.swap(curU_); ++f_; stats_.frames = double(f_); stats_.last_step = double(step);
        return wrote;
    }
    const Stats& stats() const { return stats_; }

private:
    V3 micz(V3 a) const {
        a = mic(a, box_[1] - box_[0], box_[3] - box_[2]);
        if (pz_) { const double lz = box_[5] - box_[4]; a.z -= lz * std::nearbyint(a.z / lz); }
        return a;
    }
    double zref_now() const {
        if (ref_mode_ == 0) return 0.0;
        if (ref_mode_ == 3) return ref_fixed_;
        double ext = ref_mode_ == 2 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        for (int i : ref_idx_) ext = ref_mode_ == 2 ? std::min(ext, curRaw_[i].z) : std::max(ext, curRaw_[i].z);
        double sum = 0; int n = 0;
        for (int i : ref_idx_) {
            const double z = curRaw_[i].z;
            if (ref_mode_ == 4 || (ref_mode_ == 1 && z >= ext - ref_width_) || (ref_mode_ == 2 && z <= ext + ref_width_)) { sum += z; ++n; }
        }
        need(n > 0, "reference layer selection is empty");
        return sum / n;
    }
    int window(double zrel) const {
        for (int w = 0; w < W_; ++w) if (zrel >= edges_[2 * w] && zrel < edges_[2 * w + 1]) return w;
        return W_;
    }
    int acc_group(int j, int nh) const {
        return acc_table_[size_t(acc_row_[O_[j]]) * (nh_max_ + 1) + std::min(nh, nh_max_)];
    }
    void add(int g, int w, V3 v) { double* c = &dec_buf_[(size_t(g) * (W_ + 1) + w) * 3]; c[0] += v.x; c[1] += v.y; c[2] += v.z; }
    int row_of(int j) const { return acc_row_[O_[j]]; }
    float zrel_mid(int i) const { return float(0.5 * (prevU_[i].z + curU_[i].z) - 0.5 * (zref_prev_ + zref_cur_)); }
    void record_events(const Assign& a, int64_t f) {
        for (size_t h = 0; h < H_.size(); ++h) {
            const int o0 = prevOwner_[h], o1 = a.owner[h];
            if (o0 == o1) continue;
            EventRec e{}; e.frame = f; e.h_id = ids_[H_[h]];
            e.donor_id = o0 < 0 ? -1 : ids_[O_[o0]]; e.acceptor_id = o1 < 0 ? -1 : ids_[O_[o1]];
            e.donor_row = int8_t(o0 < 0 ? -1 : row_of(o0)); e.acceptor_row = int8_t(o1 < 0 ? -1 : row_of(o1));
            e.donor_nh0 = int16_t(o0 < 0 ? -1 : prevNH_[o0]); e.donor_nh1 = int16_t(o0 < 0 ? -1 : a.nh[o0]);
            e.acceptor_nh0 = int16_t(o1 < 0 ? -1 : prevNH_[o1]); e.acceptor_nh1 = int16_t(o1 < 0 ? -1 : a.nh[o1]);
            e.donor_zrel = o0 < 0 ? std::numeric_limits<float>::quiet_NaN() : zrel_mid(O_[o0]);
            e.acceptor_zrel = o1 < 0 ? std::numeric_limits<float>::quiet_NaN() : zrel_mid(O_[o1]);
            if (o0 >= 0 && o1 >= 0) {
                const V3 hop = micz(((curU_[O_[o1]] + prevU_[O_[o1]]) - (curU_[O_[o0]] + prevU_[O_[o0]])) * 0.5);
                e.hop[0] = float(hop.x); e.hop[1] = float(hop.y); e.hop[2] = float(hop.z);
            }
            e.d_new = o1 < 0 ? -1.0f : float(nrm(bondNow_[h]));
            events_.push_back(e);
        }
    }
    void record_frame(int64_t f) {   // current nH is prevNH_ (already swapped), current positions curRaw_/curU_
        if (f % counts_stride_ == 0) {
            const size_t base = counts_.size(); counts_.resize(base + size_t(G_) * (W_ + 1), 0); count_frames_.push_back(f);
            for (size_t j = 0; j < O_.size(); ++j) {
                const int w = window(curRaw_[O_[j]].z - zref_cur_);
                ++counts_[base + size_t(acc_group(int(j), prevNH_[j])) * (W_ + 1) + w];
            }
        }
        if (f % sites_stride_ == 0) {
            for (size_t j = 0; j < O_.size(); ++j) {
                const int row = row_of(int(j)), nh = std::min(prevNH_[j], nh_max_);
                if (!tracked_[size_t(row) * (nh_max_ + 1) + nh]) continue;
                const V3& u = curU_[O_[j]];
                sites_.push_back(SiteRec{f, ids_[O_[j]], int8_t(row), int8_t(prevNH_[j]), {float(u.x), float(u.y), float(u.z)}});
            }
        }
    }
    // Same terms as motion/transfer/image above, each placed in a (group, window) cell; cells must sum to dM_site.
    void decompose(const Assign& a, V3 dMsite, double* out) {
        std::fill(dec_buf_.begin(), dec_buf_.end(), 0.0);
        const double zr = 0.5 * (zref_prev_ + zref_cur_);
        auto mid = [&](int i) { return window(0.5 * (prevU_[i].z + curU_[i].z) - zr); };
        for (size_t j = 0; j < O_.size(); ++j) {
            const int o = O_[j]; const V3 du = curU_[o] - prevU_[o]; const int w = mid(o);
            add(acc_group(int(j), prevNH_[j]), w, du * (0.5 * (qacc_[j] + prevNH_[j])));
            add(acc_group(int(j), a.nh[j]), w, du * (0.5 * (qacc_[j] + a.nh[j])));
        }
        for (int i : Mg_) add(other_group_[i], mid(i), (curU_[i] - prevU_[i]) * qf_[i]);
        for (size_t h = 0; h < H_.size(); ++h) {
            const int o0 = prevOwner_[h], o1 = a.owner[h], hi = H_[h];
            const double free0 = o0 < 0 ? 1.0 : 0.0, free1 = o1 < 0 ? 1.0 : 0.0;
            if (free0 || free1) add(g_free_, mid(hi), (curU_[hi] - prevU_[hi]) * (0.5 * (free0 + free1)));
            const int node0 = o0 < 0 ? hi : O_[o0], node1 = o1 < 0 ? hi : O_[o1];
            // all whole-cell bookkeeping of one H goes to one cell, so the parts of a hop cancel there
            V3 img{}; int wimg = mid(hi);
            if (o0 != o1) {
                const double zm = 0.25 * (prevU_[node0].z + curU_[node0].z + prevU_[node1].z + curU_[node1].z) - zr;
                // physical hop = minimum-image vector between the two sites; the remainder (whole cell vectors from
                // different unwrap histories of the two sites) is bookkeeping
                const V3 raw = ((curU_[node1] + prevU_[node1]) - (curU_[node0] + prevU_[node0])) * 0.5, hop = micz(raw);
                wimg = window(zm);
                add(g_transfer_, wimg, hop);
                img += raw - hop;
            }
            if (o0 >= 0) img -= (prevU_[hi] - prevBond_[h] - prevU_[O_[o0]]);
            if (o1 >= 0) img += (curU_[hi] - bondNow_[h] - curU_[O_[o1]]);
            add(g_image_, wimg, img);
        }
        V3 tot{};
        for (size_t c = 0; c < dec_buf_.size() / 3; ++c) { tot.x += dec_buf_[3 * c]; tot.y += dec_buf_[3 * c + 1]; tot.z += dec_buf_[3 * c + 2]; }
        need(maxabs(tot - dMsite) <= 1e-7 * (1.0 + maxabs(dMsite)), "species/window decomposition does not sum to dM_site");
        if (out) std::copy(dec_buf_.begin(), dec_buf_.end(), out);
    }
    // Exact owner selection, identical arithmetic for both search paths: the candidate set within 1.5 + 1e-8 A,
    // dmin, the tie count and the smallest-ID choice do not depend on the order in which O are offered.
    template <class Visit> void assign_into(Assign& a, Visit offer_candidates) {
        a.owner.assign(H_.size(), -1); a.nh.assign(O_.size(), 0); a.ties = 0; a.assigned = 0; a.maxdist = 0;
        std::vector<std::pair<double, int>>& cand = cand_;
        for (size_t h = 0; h < H_.size(); ++h) {
            cand.clear(); double dmin = std::numeric_limits<double>::infinity(); V3 hp = curRaw_[H_[h]];
            offer_candidates(h, [&](int oi) { V3 d = micz(hp - curRaw_[oi]); double d0 = nrm(d); if (d0 <= cut_hi_) { cand.emplace_back(d0, oi); if (d0 < dmin) dmin = d0; } });
            if (dmin <= cutoff_) {
                int oiBest = -1, nt = 0; int32_t idBest = std::numeric_limits<int32_t>::max();
                for (auto [d0, oi] : cand) if (d0 <= cutoff_ && d0 <= dmin + tol_) { ++nt; if (ids_[oi] < idBest) { idBest = ids_[oi]; oiBest = oi; } }
                need(oiBest >= 0, "tie selection failed"); int ord = oordinal_[oiBest]; need(ord >= 0, "oxygen ordinal map failed");
                a.owner[h] = ord; ++a.nh[ord]; ++a.assigned; if (nt > 1) ++a.ties; a.maxdist = std::max(a.maxdist, dmin);
            }
        }
    }
    void assign_full(Grid& g, Assign& a) {
        g.build(curRaw_, O_);
        assign_into(a, [&](size_t h, auto&& fn) { g.visit(curRaw_[H_[h]], O_, fn); });
    }
    // Verlet list: every O within r_list = 1.5 + 1e-8 + skin of each H at build time. Valid while
    // max|dU_H| + max|dU_O| (unwrapped displacement since the build) <= skin: an O that is now within the cutoff
    // was then within cutoff + dH + dO <= r_list, so the offered set always contains every true candidate.
    bool list_expired() const {
        double mh = 0, mo = 0;
        for (int i : H_) { V3 d = curU_[i] - nlU_[i]; mh = std::max(mh, d.x * d.x + d.y * d.y + d.z * d.z); }
        for (int i : O_) { V3 d = curU_[i] - nlU_[i]; mo = std::max(mo, d.x * d.x + d.y * d.y + d.z * d.z); }
        return std::sqrt(mh) + std::sqrt(mo) > nl_skin_ * (1.0 - 1e-9);
    }
    void list_build() {
        const double rlist = cut_hi_ + nl_skin_, r2 = rlist * rlist * (1.0 + 1e-9);
        nl_grid_.build(curRaw_, O_);
        nl_start_.assign(H_.size() + 1, 0); nl_.clear();
        for (size_t h = 0; h < H_.size(); ++h) {
            V3 hp = curRaw_[H_[h]];
            nl_grid_.visit(hp, O_, [&](int oi) { V3 d = micz(hp - curRaw_[oi]); if (d.x * d.x + d.y * d.y + d.z * d.z <= r2) nl_.push_back(oi); });
            nl_start_[h + 1] = int(nl_.size());
        }
        nlU_ = curU_; nl_valid_ = true;
    }
    Assign& assign(Grid& g) {
        Assign& a = cur_;
        if (!nl_enabled_ || nl_cooldown_ > 0) {
            if (nl_cooldown_ > 0 && --nl_cooldown_ == 0) nl_valid_ = false;
            assign_full(g, a); stats_.nl_full_frames += 1; return a;
        }
        bool rebuilt = false;
        if (!nl_valid_ || list_expired()) { list_build(); rebuilt = true; stats_.nl_rebuilds += 1; }
        assign_into(a, [&](size_t h, auto&& fn) { for (int k = nl_start_[h]; k < nl_start_[h + 1]; ++k) fn(nl_[k]); });
        stats_.nl_list_frames += 1;
        if (nl_audit_every_ > 0 && f_ % nl_audit_every_ == 0) {   // independent full search must agree exactly
            assign_full(g, audit_);
            need(audit_.owner == a.owner && audit_.nh == a.nh && audit_.ties == a.ties && audit_.assigned == a.assigned && audit_.maxdist == a.maxdist,
                 "neighbour-list audit failed: list and full search disagree");
            stats_.nl_audits += 1;
        }
        // adaptive fallback: if most recent frames needed a rebuild (e.g. 40 fs sampling), use the full search for a while
        nl_window_rebuilds_ += rebuilt ? 1 : 0;
        if (++nl_window_frames_ == 64) {
            if (nl_window_rebuilds_ > 32) { nl_cooldown_ = 4096; stats_.nl_fallbacks += 1; }
            nl_window_frames_ = 0; nl_window_rebuilds_ = 0;
        }
        return a;
    }
public:
    void configure_list(bool enabled, double skin, int audit_every) {
        need(skin > 0 && skin < 5, "neighbour-list skin must be in (0, 5) A"); need(audit_every >= 0, "audit interval must be >= 0");
        nl_enabled_ = enabled; nl_skin_ = skin; nl_audit_every_ = audit_every; nl_valid_ = false; nl_cooldown_ = 0;
        nl_grid_ = Grid(box_[0], box_[2], box_[4], box_[1] - box_[0], box_[3] - box_[2], box_[5] - box_[4], int(O_.size()), cut_hi_ + skin, pz_);
    }
private:
    int nat_; int64_t dstep_; double dt_, t0_; int64_t firststep_; int64_t f_ = 0; int64_t qformal_ = 0;
    std::vector<int32_t> ids_; std::vector<int8_t> role_; std::vector<double> qf_, qacc_; double box_[6];
    bool pz_ = false; double cutoff_ = 1.5, tol_ = 1e-8, cut_hi_ = 1.5 + 1e-8; int64_t qsite_base_ = 0;
    std::vector<int> H_, O_, Mg_, oordinal_;
    std::vector<V3> prevRaw_, prevU_, curRaw_, curV_, curU_, prevBond_;
    std::vector<int> prevOwner_, prevNH_; int prevAssigned_ = 0, prevTies_ = 0;
    Stats stats_;
    // reused per-frame buffers (no per-frame allocation)
    Grid grid_{0, 0, 0, 1, 1, 1, 0};
    Assign cur_;
    std::vector<std::pair<double, int>> cand_;
    std::vector<int> flow_;
    std::vector<V3> bondNow_;
    bool had_vel_ = false;
    // neighbour list state
    bool nl_enabled_ = false, nl_valid_ = false; double nl_skin_ = 1.0; int nl_audit_every_ = 1000;
    int nl_cooldown_ = 0, nl_window_frames_ = 0, nl_window_rebuilds_ = 0;
    Grid nl_grid_{0, 0, 0, 1, 1, 1, 0};
    std::vector<int> nl_start_, nl_; std::vector<V3> nlU_; Assign audit_;
    // decomposition state
    bool dec_ = false; int ref_mode_ = 0; double ref_width_ = 0, ref_fixed_ = 0, zref_prev_ = 0, zref_cur_ = 0;
    std::vector<int> ref_idx_; int nh_max_ = 0; std::vector<int32_t> acc_table_, acc_row_, other_group_;
    int G_ = 0, W_ = 0, g_free_ = 0, g_transfer_ = 0, g_image_ = 0; std::vector<double> edges_, dec_buf_;
    bool ev_ = false; int counts_stride_ = 10, sites_stride_ = 10; std::vector<uint8_t> tracked_;
};

// ---- streaming LAMMPS dump reader (orthogonal box; id type x y z [vx vy vz]) ----
int split(std::string_view s, std::string_view* out, int cap) {
    int n = 0; size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r')) ++i; if (i == s.size()) break;
        size_t j = i; while (j < s.size() && s[j] != ' ' && s[j] != '\t' && s[j] != '\r') ++j;
        if (n < cap) out[n] = s.substr(i, j - i); ++n; i = j;
    }
    return n;
}
template <class T> T num(std::string_view s, const char* w) { T v{}; auto r = std::from_chars(s.data(), s.data() + s.size(), v); need(r.ec == std::errc() && r.ptr == s.data() + s.size(), std::string("bad ") + w); return v; }
double real(std::string_view s, const char* w) { double x = num<double>(s, w); need(std::isfinite(x), std::string("nonfinite ") + w); return x; }

class LammpsReader {
public:
    explicit LammpsReader(const char* path) : f_(std::fopen(path, "rb")), b_(64u * 1024u * 1024u) { need(f_ != nullptr, std::string("cannot open ") + path); std::setvbuf(f_, nullptr, _IOFBF, 64u * 1024u * 1024u); }
    ~LammpsReader() { if (f_) std::fclose(f_); }
    // Returns false at a clean end of file.
    bool frame(int nat_expected, std::vector<int32_t>& ids, std::vector<int16_t>& types, double* pos, double* vel, int64_t& step, double* bb, bool& has_vel) {
        std::string_view s;
        for (;;) { if (!line(s)) return false; bool white = true; for (char c : s) if (c != ' ' && c != '\t') white = false; if (!white) break; }
        need(s == "ITEM: TIMESTEP", "expected timestep marker"); need(line(s), "truncated step"); step = num<int64_t>(s, "step");
        need(line(s) && s == "ITEM: NUMBER OF ATOMS", "missing atom count header"); need(line(s), "truncated atom count");
        int nat = num<int>(s, "atom count"); need(nat == nat_expected, "atom count changed");
        need(line(s), "truncated box header"); need(s.substr(0, 16) == "ITEM: BOX BOUNDS", "missing box header");
        need(s.find("xy") == std::string_view::npos && s.find("xz") == std::string_view::npos && s.find("yz") == std::string_view::npos, "triclinic dump unsupported");
        std::string_view tok[4];
        for (int k = 0; k < 3; ++k) { need(line(s), "truncated box bounds"); int c = split(s, tok, 4); need(c == 2, "triclinic bounds row"); bb[2 * k] = real(tok[0], "box bound"); bb[2 * k + 1] = real(tok[1], "box bound"); need(bb[2 * k + 1] > bb[2 * k], "invalid box"); }
        need(line(s), "missing atom header"); need(s.substr(0, 11) == "ITEM: ATOMS", "missing ATOMS marker");
        std::string_view ht[32]; int hn = split(s, ht, 32); need(hn <= 32, "too many atom columns"); int ncol = hn - 2;
        const char* names[] = {"id", "type", "x", "y", "z", "vx", "vy", "vz"}; int ci[8];
        for (int k = 0; k < 8; ++k) { ci[k] = -1; for (int j = 2; j < hn; ++j) if (ht[j] == names[k]) ci[k] = j - 2; if (k < 5) need(ci[k] >= 0, std::string("missing column ") + names[k]); }
        bool hv = ci[5] >= 0 && ci[6] >= 0 && ci[7] >= 0;
        if (!init_) { has_vel_ = hv; } else need(hv == has_vel_, "velocity columns changed");
        has_vel = has_vel_;
        std::string_view a[32];
        if (!init_) { ids.resize(nat); types.resize(nat); ix_.reserve(nat * 2); seen_.assign(nat, 0); }
        else std::fill(seen_.begin(), seen_.end(), 0);
        for (int k = 0; k < nat; ++k) {
            need(line(s), "truncated atom rows"); need(split(s, a, 32) == ncol, "atom row field count");
            int32_t id = num<int32_t>(a[ci[0]], "atom ID"); int16_t t = num<int16_t>(a[ci[1]], "atom type");
            need(id > 0, "nonpositive atom ID"); need(t >= 1 && t <= 3, "unknown atom type");
            int i;
            if (!init_) { need(ix_.emplace(id, k).second, "duplicate first-frame ID"); ids[k] = id; types[k] = t; i = k; }
            else { auto it = ix_.find(id); need(it != ix_.end(), "atom ID set changed"); i = it->second; need(!seen_[i], "duplicate atom ID"); seen_[i] = 1; need(t == types[i], "atom type changed"); }
            pos[3 * i] = real(a[ci[2]], "x"); pos[3 * i + 1] = real(a[ci[3]], "y"); pos[3 * i + 2] = real(a[ci[4]], "z");
            if (vel) { if (hv) { vel[3 * i] = real(a[ci[5]], "vx"); vel[3 * i + 1] = real(a[ci[6]], "vy"); vel[3 * i + 2] = real(a[ci[7]], "vz"); } else { vel[3 * i] = vel[3 * i + 1] = vel[3 * i + 2] = 0.0; } }
        }
        if (init_) for (int z : seen_) need(z == 1, "missing atom ID");
        init_ = true; return true;
    }

private:
    bool refill() { if (eof_) return false; n_ = std::fread(b_.data(), 1, b_.size(), f_); p_ = 0; if (n_) return true; if (std::ferror(f_)) throw std::runtime_error("read error"); eof_ = true; return false; }
    bool line(std::string_view& out) {
        spill_.clear();
        for (;;) {
            if (p_ >= n_ && !refill()) { if (spill_.empty()) return false; out = spill_; if (!out.empty() && out.back() == '\r') out.remove_suffix(1); return true; }
            const char* s = b_.data() + p_; const void* q = std::memchr(s, '\n', n_ - p_);
            if (q) { const char* e = (const char*)q; size_t m = e - s; p_ += m + 1; if (spill_.empty()) out = std::string_view(s, m); else { spill_.append(s, m); out = spill_; } if (!out.empty() && out.back() == '\r') out.remove_suffix(1); return true; }
            spill_.append(s, n_ - p_); p_ = n_;
        }
    }
    FILE* f_; std::vector<char> b_; size_t p_ = 0, n_ = 0; bool eof_ = false; std::string spill_;
    bool init_ = false, has_vel_ = false; std::unordered_map<int32_t, int> ix_; std::vector<int> seen_;
};

struct LmpHandle { LammpsReader reader; std::vector<int32_t> ids; std::vector<int16_t> types; explicit LmpHandle(const char* p) : reader(p) {} };

}  // namespace

extern "C" {

const char* waterint_sc_error() { return g_error.c_str(); }

void* waterint_sc_create2(int nat, const int32_t* ids, const int8_t* role, const double* qf, const double* box6, const uint8_t* pbc3,
                          double cutoff, double tie_tol, int64_t first_step, int64_t dstep, double dt_ps, double t0_ps) {
    try { return new SiteCurrent(nat, ids, role, qf, box6, pbc3, cutoff, tie_tol, first_step, dstep, dt_ps, t0_ps); }
    catch (const std::exception& e) { g_error = e.what(); return nullptr; }
}

// MgO convenience (types 1 = H, 2 = Mg, 3 = O; numbers of the mgo_water profile).
void* waterint_sc_create(int nat, const int16_t* types, const int32_t* ids, const double* box6, int64_t first_step, int64_t dstep, double dt_ps, double t0_ps) {
    try {
        std::vector<int8_t> role(nat); std::vector<double> qf(nat); const uint8_t pbc[3] = {1, 1, 0};
        for (int i = 0; i < nat; ++i) {
            need(types[i] >= 1 && types[i] <= 3, "unknown atom type (1=H, 2=Mg, 3=O)");
            role[i] = types[i] == 1 ? 1 : (types[i] == 3 ? 2 : 0); qf[i] = types[i] == 1 ? 1.0 : (types[i] == 2 ? 2.0 : -2.0);
        }
        return new SiteCurrent(nat, ids, role.data(), qf.data(), box6, pbc, 1.5, 1e-8, first_step, dstep, dt_ps, t0_ps);
    } catch (const std::exception& e) { g_error = e.what(); return nullptr; }
}

int waterint_sc_set_decomposition(void* h, const uint8_t* ref_mask, int ref_mode, double ref_width, double ref_fixed, const int32_t* acc_row,
                                  int n_rows, int nh_max, const int32_t* table, const int32_t* other_group, int free_h_group, int transfer_group,
                                  int image_group, int n_groups, int n_windows, const double* edges) {
    try {
        static_cast<SiteCurrent*>(h)->set_decomposition(ref_mask, ref_mode, ref_width, ref_fixed, acc_row, n_rows, nh_max, table, other_group,
                                                        free_h_group, transfer_group, image_group, n_groups, n_windows, edges);
        return 0;
    } catch (const std::exception& e) { g_error = e.what(); return 1; }
}

void waterint_sc_free(void* h) { delete static_cast<SiteCurrent*>(h); }

// pos: nf*nat*3; vel: nf*nat*3 or null; steps: nf; boxes: nf*6 or null (fixed box).
// frames_out: nf records; intervals_out: room for nf records; *n_intervals receives the count written.
// dec_out (optional): room for nf * cells * 3 doubles, filled for every interval written (cells = G * (W + 1));
// zref_out (optional): nf doubles (reference plane height of every frame).
int waterint_sc_push(void* h, int64_t nf, int nat, const double* pos, const double* vel, const int64_t* steps, const double* boxes,
                     void* frames_out, void* intervals_out, int64_t* n_intervals, double* dec_out, double* zref_out) {
    try {
        auto* sc = static_cast<SiteCurrent*>(h); auto* fr = static_cast<FrameRec*>(frames_out); auto* ir = static_cast<IntervalRec*>(intervals_out);
        int64_t ni = 0; const size_t cells3 = size_t(sc->decomposition_cells()) * 3;
        for (int64_t k = 0; k < nf; ++k) {
            const size_t off = size_t(k) * size_t(nat) * 3;
            double* d = dec_out && cells3 ? dec_out + size_t(ni) * cells3 : nullptr;
            if (sc->push(pos + off, vel ? vel + off : nullptr, steps[k], boxes ? boxes + 6 * k : nullptr, fr[k], ir[ni], d, zref_out ? zref_out + k : nullptr)) ++ni;
        }
        *n_intervals = ni; return 0;
    } catch (const std::exception& e) { g_error = e.what(); return 1; }
}

int waterint_sc_configure(void* h, int use_list, double skin, int audit_every) {
    try { static_cast<SiteCurrent*>(h)->configure_list(use_list != 0, skin, audit_every); return 0; }
    catch (const std::exception& e) { g_error = e.what(); return 1; }
}

int waterint_sc_set_events(void* h, int enable, int counts_stride, int sites_stride, const uint8_t* tracked) {
    try { static_cast<SiteCurrent*>(h)->set_events(enable != 0, counts_stride, sites_stride, tracked); return 0; }
    catch (const std::exception& e) { g_error = e.what(); return 1; }
}

// Pending record counts: out[0] events, out[1] count rows, out[2] tracked sites.
void waterint_sc_pending(void* h, int64_t* out) {
    auto* sc = static_cast<SiteCurrent*>(h);
    out[0] = int64_t(sc->events_.size()); out[1] = int64_t(sc->count_frames_.size()); out[2] = int64_t(sc->sites_.size());
}

// Copy and clear all pending records (buffers sized from waterint_sc_pending; counts: rows * cells int32).
void waterint_sc_take(void* h, void* events_out, int64_t* count_frames_out, int32_t* counts_out, void* sites_out) {
    auto* sc = static_cast<SiteCurrent*>(h);
    if (events_out && !sc->events_.empty()) std::memcpy(events_out, sc->events_.data(), sc->events_.size() * sizeof(EventRec));
    if (count_frames_out && !sc->count_frames_.empty()) std::memcpy(count_frames_out, sc->count_frames_.data(), sc->count_frames_.size() * sizeof(int64_t));
    if (counts_out && !sc->counts_.empty()) std::memcpy(counts_out, sc->counts_.data(), sc->counts_.size() * sizeof(int32_t));
    if (sites_out && !sc->sites_.empty()) std::memcpy(sites_out, sc->sites_.data(), sc->sites_.size() * sizeof(SiteRec));
    sc->events_.clear(); sc->count_frames_.clear(); sc->counts_.clear(); sc->sites_.clear();
}

int waterint_sc_stats(void* h, double* out19) {
    const Stats& s = static_cast<SiteCurrent*>(h)->stats();
    const double v[19] = {s.max_box_drift, s.max_step_comp, s.max_step_norm, s.max_vel, s.max_vel_ratio, s.crossed_x, s.crossed_y,
                          s.frames, s.first_step, s.last_step, s.formal_q, s.owner_changes, s.ties, s.free_h_samples,
                          s.nl_list_frames, s.nl_full_frames, s.nl_rebuilds, s.nl_audits, s.nl_fallbacks};
    std::memcpy(out19, v, sizeof(v)); return 0;
}

void* waterint_lmp_open(const char* path) {
    try { return new LmpHandle(path); } catch (const std::exception& e) { g_error = e.what(); return nullptr; }
}
void waterint_lmp_close(void* h) { delete static_cast<LmpHandle*>(h); }

// Reads up to nmax frames. ids/types (nat) are filled from the first frame of the file and positions are
// returned in that first-frame order. *nread = frames read (0 at end of file). Returns 0, or 1 on error.
int waterint_lmp_read(void* h, int64_t nmax, int nat, int32_t* ids, int16_t* types, double* pos, double* vel, int64_t* steps, double* boxes, int64_t* nread, int* has_vel) {
    try {
        auto* lh = static_cast<LmpHandle*>(h); int64_t k = 0; bool hv = false;
        for (; k < nmax; ++k) {
            const size_t off = size_t(k) * size_t(nat) * 3;
            if (!lh->reader.frame(nat, lh->ids, lh->types, pos + off, vel ? vel + off : nullptr, steps[k], boxes + 6 * k, hv)) break;
        }
        if (!lh->ids.empty()) { std::memcpy(ids, lh->ids.data(), sizeof(int32_t) * nat); std::memcpy(types, lh->types.data(), sizeof(int16_t) * nat); }
        *nread = k; *has_vel = hv ? 1 : 0; return 0;
    } catch (const std::exception& e) { g_error = e.what(); return 1; }
}

}  // extern "C"
