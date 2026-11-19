// hb_correlation.cpp - streaming one-pass kernel: seeded geometric C_PT / C_HB / proton-sharing FES, O-centred species
// density, O-H orientation and per-frame species counts, donated / accepted H-bond statistics and fixed-identity xy MSD at
// sparse lags, all per block of BLK frames (definitions: analysis/dynamics/hb_correlation_core.py). Identical counts to
// the GPU kernel (backends/gpu/hb_correlation_gpu.cu). Frames are pushed in time order with push(); every W frames
// a window is processed (owners, structure, seeds, MSD), seeds are correlated once their 100 ps lag range is in the ring.
// OpenMP (optional): parallel over frames and seeds; results do not depend on the thread count.
// Atom input: O and H only (types 3 / 1), in the caller's order; atom IDs only break ties (lowest ID wins).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

namespace waterint_hbc {
namespace fs = std::filesystem;
typedef unsigned long long u64;
static std::string g_error;
static void need(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
template <class T> void writevec(const fs::path& p, const std::vector<T>& v) {
  std::ofstream f(p, std::ios::binary); need(bool(f), "cannot write " + p.string());
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T)); need(bool(f), "write failed");
}
struct Seed { int32_t f, h, d, a, g, ch; };
enum { S_UNASSIGNED, S_OWNER_TIES, S_PRIMARY_TIES, S_ELIG0, S_ELIG1, S_PRIM0, S_PRIM1, S_FESX0, S_FESX1, S_FESX2, S_FESX3,
       S_FESO0, S_FESO1, S_FESO2, S_FESO3, S_REG0, S_SPC0 = S_REG0 + 6, S_TS0 = S_SPC0 + 5, S_NSTAT = S_TS0 + 4 };
static inline void mic(double& x, double& y, double lx, double ly) { x -= lx * std::rint(x / lx); y -= ly * std::rint(y / ly); }
static inline int direction(int rd, int cd, int ra, int ca) {
  if (rd == 1 && cd == 1 && ra == 1 && ca == 2) return 0;
  if (rd == 1 && cd == 2 && ra == 2 && ca == 1) return 1;
  if (rd == 1 && cd == 2 && ra == 1 && ca == 1) return 2;
  if (rd == 2 && cd == 2 && ra == 1 && ca == 1) return 3;
  return -1;
}
static inline int binof(double x, double lo, double step, int n) { int b = (int)std::floor((x - lo) / step); return b >= 0 && b < n ? b : -1; }
static inline int binc(double x, double lo, double hi, int n) { if (x < lo || x > hi) return -1; if (x == hi) return n - 1; return (int)std::floor((x - lo) / (hi - lo) * n); }
static inline int och(int lat, int reg, int cnt) { const int kind = std::min(cnt, 4); if (lat) return kind; const int r = reg == 1 ? 0 : (reg == 2 ? 1 : (reg == 3 ? 2 : 3)); return 5 + r * 5 + kind; }
static inline bool strict_ok(double r, double c) { return r >= 2.2 && r < 3.2 && c <= -0.86602540378443864676; }
template <class T> static inline void ainc(T& x, T v = 1) {
#pragma omp atomic
  x += v;
}
static inline int wrapcell(double x, double length, double width, int n) { double y = x - length * std::floor(x / length); return std::min(n - 1, int(y / width)); }

struct Work {
  std::vector<int> cnt, reg, reg2, acc, head, next; std::vector<Seed> seeds, samples; u64 stats[S_NSTAT] = {}; u64 seedden[8] = {}; double minoo = 1e10;
  struct Cand { int a; double r, c, ha; bool st; }; std::vector<Cand> cand; std::vector<short> own;
};

struct HBC {
  static constexpr int LMAX = 25000;
  int na, no, nh, nf, W, BLK, MS, MAXLAG, NB, nblk, nl, norig, nsf, nnl, nm, nml, MR, nthr, nx, ny;
  double lx, ly, wx, wy, wz = 3.5; size_t fd, nmq;
  std::vector<int32_t> ids; std::vector<int> os, hs, oidO, nlo, matom; std::vector<unsigned char> lattice;
  std::vector<int32_t> lags, mlags; std::vector<int> sampleframes;
  std::vector<double> pos, zref, zref2; std::vector<unsigned short> owners;
  std::vector<u64> stats, seedden, origin_den, hist, testhist, dout, oout, pop, hbd, hba, numb, mcnt;
  std::vector<unsigned int> bhist, dens, ori; std::vector<short> tspec;
  std::vector<double> mu, mnum, mprev, mimg; std::vector<unsigned char> mcode; double mL[2];
  std::vector<Work> work; std::vector<Seed> pending; u64 nseed_total = 0;
  long Fin = 0, F = 0; double tframe = 0, tcorr = 0, tmsd = 0; std::chrono::steady_clock::time_point t0;

  HBC(int na_, const int32_t* ids_, const int16_t* types, const uint8_t* lat, long nftot, double lx_, double ly_, int W_, int BLK_, int MS_, int MAXLAG_)
      : na(na_), nf((int)nftot), W(W_), BLK(BLK_), MS(MS_), MAXLAG(MAXLAG_), lx(lx_), ly(ly_) {
    need(nftot > LMAX && nftot < 2000000000L && W >= 50 && BLK >= 50, "need more than 25000 frames, W >= 50, BLK >= 50");
    need(MS >= 1 && MAXLAG >= 1, "bad MSD stride / max lag");
    NB = W + LMAX; nblk = (nf - 1) / BLK + 1; t0 = std::chrono::steady_clock::now();
    ids.assign(ids_, ids_ + na);
    for (int i = 0; i < na; ++i) { if (types[i] == 3) os.push_back(i); else if (types[i] == 1) hs.push_back(i); else need(false, "atoms must be O (3) or H (1) only"); }
    no = os.size(); nh = hs.size(); need(no < 32000 && no > 0 && nh > 0, "bad O / H counts");
    lattice.resize(no); oidO.resize(no); for (int o = 0; o < no; ++o) { lattice[o] = lat[os[o]] ? 1 : 0; oidO[o] = ids[os[o]]; }
    for (int i = 0; i <= 250; ++i) lags.push_back(i); for (int i = 260; i <= 2500; i += 10) lags.push_back(i); for (int i = 2550; i <= 25000; i += 50) lags.push_back(i);
    nl = lags.size(); norig = (nf - 25001) / 50 + 1;
    sampleframes = {0, 1, 50, 250, 2500, 12500, 25000, nf / 2, nf - 25001, nf - 1};
    std::sort(sampleframes.begin(), sampleframes.end()); sampleframes.erase(std::unique(sampleframes.begin(), sampleframes.end()), sampleframes.end());
    fd = (size_t)na * 3;
    pos.resize((size_t)NB * fd); zref.resize(NB); zref2.resize(NB); owners.resize((size_t)NB * nh);
    stats.assign(S_NSTAT, 0); seedden.assign(8, 0); origin_den.assign(8 * (size_t)norig, 0); hist.assign((size_t)2 * 2 * 2 * 280 * 350, 0); testhist.assign((size_t)2 * 2 * 280 * 350, 0);
    bhist.assign((size_t)nblk * 4 * 280 * 350, 0); dens.assign((size_t)nblk * 25 * 1240, 0); ori.assign((size_t)nblk * 5 * 360 * 360, 0);
    dout.assign((size_t)nblk * 25, 0); oout.assign((size_t)nblk * 5, 0); pop.assign((size_t)nblk * 25, 0); hbd.assign((size_t)nblk * 200, 0); hba.assign((size_t)nblk * 400, 0); numb.assign((size_t)nblk * 16 * nl, 0);
    tspec.assign((size_t)nf * 4, 0);
    nsf = (nf - 1) / MS + 1; nlo.assign(no, -1);
    for (int o = 0; o < no; ++o) if (!lattice[o]) { nlo[o] = (int)matom.size(); matom.push_back(os[o]); }
    nnl = (int)matom.size(); for (int t = 0; t < nh; ++t) matom.push_back(hs[t]); nm = (int)matom.size();
    { const int top = std::min(MAXLAG, nsf - 1);
      for (int i = 0; i <= 100 && i <= top; ++i) mlags.push_back(i); for (int i = 110; i <= 1000 && i <= top; i += 10) mlags.push_back(i);
      for (int i = 1100; i <= 10000 && i <= top; i += 100) mlags.push_back(i); for (int i = 11000; i <= top; i += 1000) mlags.push_back(i); }
    nml = (int)mlags.size(); MR = std::min(nsf, mlags.back() + W / MS + 2); nmq = (size_t)nblk * 14 * nml;
    mu.assign((size_t)MR * nm * 2, 0.0); mnum.assign(2 * nmq, 0.0); mprev.assign((size_t)nm * 2, 0.0); mimg.assign((size_t)nm * 2, 0.0); mcode.assign((size_t)MR * nm, 0);
    mcnt.assign(nmq, 0); mL[0] = lx; mL[1] = ly;
    nthr = omp_get_max_threads(); work.resize(nthr);
    nx = int(lx / 3.5); ny = int(ly / 3.5); need(nx >= 3 && ny >= 3, "cell too small"); wx = lx / nx; wy = ly / ny;
    for (auto& w : work) { w.cnt.resize(no); w.reg.resize(no); w.reg2.resize(no); w.acc.resize(2 * no); w.next.resize(no); w.cand.reserve(64); w.own.resize(nh); }
  }
  double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

  void frame(int f, Work& w) {
    const int slot = f % NB; const double* X = pos.data() + (size_t)slot * fd; const double zr = zref[slot], zr2 = zref2[slot]; const int blk = f / BLK;
    double zmin = 1e100, zmax = -1e100; for (int o = 0; o < no; ++o) { double z = X[3 * os[o] + 2]; zmin = std::min(zmin, z); zmax = std::max(zmax, z); }
    const int nz = int((zmax - zmin) / wz) + 1; w.head.assign((size_t)nx * ny * nz, -1);
    for (int o = 0; o < no; ++o) { const double* p = X + 3 * os[o]; int k = (int((p[2] - zmin) / wz) * ny + wrapcell(p[1], ly, wy, ny)) * nx + wrapcell(p[0], lx, wx, nx); w.next[o] = w.head[k]; w.head[k] = o; }
    auto around = [&](const double* p, auto&& fun) {
      const int x = wrapcell(p[0], lx, wx, nx), y = wrapcell(p[1], ly, wy, ny), z = int(std::floor((p[2] - zmin) / wz));
      for (int dz = -1; dz <= 1; ++dz) { int zz = z + dz; if (zz < 0 || zz >= nz) continue;
        for (int dy = -1; dy <= 1; ++dy) { int yy = (y + dy + ny) % ny;
          for (int dx = -1; dx <= 1; ++dx) { int xx = (x + dx + nx) % nx; for (int j = w.head[(zz * ny + yy) * nx + xx]; j >= 0; j = w.next[j]) fun(j); } } }
    };
    std::fill(w.cnt.begin(), w.cnt.end(), 0); std::fill(w.acc.begin(), w.acc.end(), 0);
    u64 ts[4] = {0, 0, 0, 0};
    for (int t = 0; t < nh; ++t) {                  // owner of each H (nearest O within 1.5 A, ties 1e-8 A -> lowest ID)
      const double* hp = X + 3 * hs[t]; double best = 1e100;
      around(hp, [&](int o) { const double* q = X + 3 * os[o]; double dx = hp[0] - q[0], dy = hp[1] - q[1], dz = hp[2] - q[2]; mic(dx, dy, lx, ly);
        double dd = dx * dx + dy * dy + dz * dz; if (dd <= 2.25 && dd < best) best = dd; });
      int owner = -1, ntie = 0;
      if (best < 1e99) {
        const double lim = std::pow(std::sqrt(best) + 1e-8, 2.0);
        around(hp, [&](int o) { const double* q = X + 3 * os[o]; double dx = hp[0] - q[0], dy = hp[1] - q[1], dz = hp[2] - q[2]; mic(dx, dy, lx, ly);
          double dd = dx * dx + dy * dy + dz * dz; if (dd <= 2.25 && dd <= lim) { ++ntie; if (owner < 0 || oidO[o] < oidO[owner]) owner = o; } });
      }
      if (ntie > 1) ++w.stats[S_OWNER_TIES];
      if (owner < 0) ++w.stats[S_UNASSIGNED]; else ++w.cnt[owner];
      w.own[t] = (short)owner; owners[(size_t)slot * nh + t] = owner < 0 ? 65535 : (unsigned short)owner;
    }
    for (int o = 0; o < no; ++o) {                  // regions, species, density
      const double zo = X[3 * os[o] + 2]; const double z = zo - zr;
      const int r = lattice[o] ? 0 : (z >= 1.5 && z < 2.8 ? 1 : (z >= 2.8 && z < 4.0 ? 2 : (z >= 4 && z < 30 ? 3 : (z < 1.5 ? 4 : 5))));
      w.reg[o] = r; ++w.stats[S_REG0 + r]; ++w.stats[S_SPC0 + std::min(w.cnt[o], 4)];
      const double z2 = zo - zr2;
      const int r2 = lattice[o] ? 0 : (z2 >= 1.5 && z2 < 2.8 ? 1 : (z2 >= 2.8 && z2 < 4.0 ? 2 : (z2 >= 4 && z2 < 30 ? 3 : (z2 < 1.5 ? 4 : 5))));
      w.reg2[o] = r2; if (f % MS == 0 && nlo[o] >= 0) mcode[(size_t)((f / MS) % MR) * nm + nlo[o]] = (unsigned char)r2;
      const size_t bc = (size_t)blk * 25 + och(lattice[o], r2, w.cnt[o]);
      ainc(pop[bc]); const int iz = binc(z2, -1.0, 30.0, 1240);
      if (iz >= 0) ainc(dens[bc * 1240 + iz]); else ainc(dout[bc]);
      if (w.cnt[o] == 1) { if (lattice[o]) ++ts[0]; else if (r2 >= 1 && r2 <= 3) ++ts[r2]; }
    }
    for (int j = 0; j < 4; ++j) { tspec[(size_t)f * 4 + j] = (short)ts[j]; w.stats[S_TS0 + j] += ts[j]; }
    if (f % MS == 0) for (int t = 0; t < nh; ++t) { const int ow = w.own[t]; mcode[(size_t)((f / MS) % MR) * nm + nnl + t] = (unsigned char)(ow < 0 ? 6 : (lattice[ow] ? 0 : w.reg2[ow])); }
    const bool origin = (f % 50 == 0) && (f + 25000 < nf);
    const bool sample = std::binary_search(sampleframes.begin(), sampleframes.end(), f);
    for (int t = 0; t < nh; ++t) {                  // eligible edges, primary acceptors, seeds, FES, orientation
      const int d = w.own[t]; if (d < 0) continue;
      const double* hp = X + 3 * hs[t]; const double* dp = X + 3 * os[d];
      double ux = dp[0] - hp[0], uy = dp[1] - hp[1], uz = dp[2] - hp[2]; mic(ux, uy, lx, ly); const double un = ux * ux + uy * uy + uz * uz;
      {
        int cho = -1; if (lattice[d]) { if (w.cnt[d] == 1) cho = 1; } else { if (w.cnt[d] == 1) cho = 0; else if (w.cnt[d] == 2) cho = 3; else if (w.cnt[d] == 3) cho = 4; }
        if (cho >= 0 && un > 0) {
          const double a = std::acos(std::fmax(-1.0, std::fmin(1.0, (-uz) / std::sqrt(un)))) * 180 / 3.14159265358979323846;
          const int iz = binc(dp[2] - zr2, -1.0, 8.0, 360), ia = binc(a, 0.0, 180.0, 360); const size_t bc = (size_t)blk * 5 + cho;
          if (iz >= 0 && ia >= 0) ainc(ori[(bc * 360 + iz) * 360 + ia]); else ainc(oout[bc]);
        }
      }
      w.cand.clear(); double best[2] = {1e100, 1e100}; u64 el0 = 0, el1 = 0;
      around(dp, [&](int a) {
        if (a == d) return; const double* ap = X + 3 * os[a];
        double rx = dp[0] - ap[0], ry = dp[1] - ap[1], rz = dp[2] - ap[2]; mic(rx, ry, lx, ly); const double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > 12.25) return;
        double vx = ap[0] - hp[0], vy = ap[1] - hp[1], vz = ap[2] - hp[2]; mic(vx, vy, lx, ly); const double vn = vx * vx + vy * vy + vz * vz;
        if (un <= 0 || vn <= 0) return;
        const double c = (ux * vx + uy * vy + uz * vz) / std::sqrt(un * vn); if (c > -.5) return;
        const double r = std::sqrt(r2), ha = std::sqrt(vn); w.minoo = std::fmin(w.minoo, r); ++el0; ++w.acc[a];
        const bool st = strict_ok(r, c); if (st) { ++el1; ++w.acc[no + a]; }
        best[0] = std::fmin(best[0], ha); if (st) best[1] = std::fmin(best[1], ha);
        w.cand.push_back({a, r, c, ha, st});
      });
      w.stats[S_ELIG0] += el0; w.stats[S_ELIG1] += el1;
      {
        const size_t bc = ((size_t)blk * 2) * 25 + och(lattice[d], w.reg[d], w.cnt[d]);
        ainc(hbd[bc * 4 + std::min((int)el0, 3)]); ainc(hbd[(bc + 25) * 4 + std::min((int)el1, 3)]);
      }
      for (int g = 0; g < 2; ++g) {
        int chosen = -1, ties = 0; double cr = 0;
        for (const auto& q : w.cand) {
          if (g == 1 && !q.st) continue;
          if (q.ha <= best[g] + 1e-8) { ++ties; if (chosen < 0 || oidO[q.a] < oidO[chosen]) { chosen = q.a; cr = q.r; } }
        }
        if (ties > 1) ++w.stats[S_PRIMARY_TIES];
        if (chosen < 0) continue;
        ++w.stats[g ? S_PRIM1 : S_PRIM0];
        const int a = chosen; const int ch = direction(w.reg[d], w.cnt[d], w.reg[a], w.cnt[a]);
        if (origin && ch >= 0) { w.seeds.push_back(Seed{f, t, d, a, g, ch}); ++w.seedden[g * 4 + ch]; ainc(origin_den[(size_t)(g * 4 + ch) * norig + f / 50]); }
        if (sample && ch >= 0) w.samples.push_back(Seed{f, t, d, a, g, ch});
        if (w.cnt[d] != 2 || w.cnt[a] != 1 || lattice[d] || lattice[a]) continue;
        int fc = -1, oa = -1, ob = -1;
        if (w.reg[d] == 1 && w.reg[a] == 1) { fc = 0; oa = oidO[d] < oidO[a] ? d : a; ob = oa == d ? a : d; }
        else if ((w.reg[d] == 1 && w.reg[a] == 2) || (w.reg[d] == 2 && w.reg[a] == 1)) { fc = 1; oa = w.reg[d] == 1 ? d : a; ob = oa == d ? a : d; }
        if (fc < 0) continue;
        ++w.stats[S_FESX0 + g * 2 + fc];
        const double* pa = X + 3 * os[oa]; const double* pb = X + 3 * os[ob];
        double ax = pa[0] - hp[0], ay = pa[1] - hp[1], az = pa[2] - hp[2]; mic(ax, ay, lx, ly);
        double bx = pb[0] - hp[0], by = pb[1] - hp[1], bz = pb[2] - hp[2]; mic(bx, by, lx, ly);
        const double delta = std::sqrt(ax * ax + ay * ay + az * az) - std::sqrt(bx * bx + by * by + bz * bz);
        const int db = binof(delta, -3.5, .025, 280), rb = binof(cr, 0, .01, 350);
        if (db < 0 || rb < 0) { ++w.stats[S_FESO0 + g * 2 + fc]; continue; }
        const int half = f < (nf - 1) / 2 ? 0 : 1;
        ainc(hist[((((size_t)g * 2 + fc) * 2 + half) * 280 + db) * 350 + rb]);
        ainc(bhist[((((size_t)blk) * 4 + g * 2 + fc) * 280 + db) * 350 + rb]);
        if (sample) ainc(testhist[(((size_t)g * 2 + fc) * 280 + db) * 350 + rb]);
      }
    }
    for (int o = 0; o < no; ++o) {                  // accepted eligible H bonds per O
      const size_t bc = ((size_t)blk * 2) * 25 + och(lattice[o], w.reg[o], w.cnt[o]);
      ainc(hba[bc * 8 + std::min(w.acc[o], 7)]); ainc(hba[(bc + 25) * 8 + std::min(w.acc[no + o], 7)]);
    }
  }

  void push(int n, const double* zr, const double* zr2, const double* p) {
    for (int i = 0; i < n; ++i) {
      need(Fin < nf, "more frames pushed than declared");
      const double* X = p + (size_t)i * fd;
      for (size_t k = 0; k < fd; ++k) need(std::isfinite(X[k]), "nonfinite coordinate");
      need(std::isfinite(zr[i]) && std::isfinite(zr2[i]), "nonfinite z_ref");
      const int slot = (int)(Fin % NB); zref[slot] = zr[i]; zref2[slot] = zr2[i]; std::memcpy(pos.data() + (size_t)slot * fd, X, sizeof(double) * fd);
      const long fg = Fin;
      if (fg % MS == 0) {                           // unwrap x, y against the previous strided frame
        double* dst = mu.data() + (size_t)((fg / MS) % MR) * nm * 2;
        for (int a = 0; a < nm; ++a) for (int ax = 0; ax < 2; ++ax) {
          const double x = X[3 * matom[a] + ax]; const size_t q = (size_t)a * 2 + ax;
          if (fg > 0) mimg[q] -= std::rint((x - mprev[q]) / mL[ax]);
          mprev[q] = x; dst[q] = x + mL[ax] * mimg[q];
        }
      }
      ++Fin;
      if (Fin - F == W) window((int)(Fin - F));
    }
  }

  void window(int n) {
    const double tb = now();
#pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; ++i) frame((int)F + i, work[omp_get_thread_num()]);
    {                                               // MSD for the strided frames of this window (origin jo = j - lag)
      const double tm = now(); const int j0 = (int)((F + MS - 1) / MS), j1 = (int)((F + n + MS - 1) / MS);
#pragma omp parallel
      {
        std::vector<double> ln(2 * nmq, 0.0); std::vector<u64> lc(nmq, 0);
#pragma omp for schedule(dynamic, 8) nowait
        for (int j = j0; j < j1; ++j) {
          const double* a = mu.data() + (size_t)(j % MR) * nm * 2;
          for (int l = 0; l < nml; ++l) {
            const int jo = j - mlags[l]; if (jo < 0) break;
            const double* b = mu.data() + (size_t)(jo % MR) * nm * 2; const unsigned char* code = mcode.data() + (size_t)(jo % MR) * nm;
            const size_t base = ((size_t)((long)jo * MS / BLK) * 14) * nml + l;
            for (int i = 0; i < nm; ++i) {
              const size_t q = base + (size_t)((i < nnl ? 0 : 7) + code[i]) * nml; const double dx = a[2 * i] - b[2 * i], dy = a[2 * i + 1] - b[2 * i + 1];
              ln[2 * q] += dx * dx; ln[2 * q + 1] += dy * dy; ++lc[q];
            }
          }
        }
#pragma omp critical
        { for (size_t i = 0; i < nmq; ++i) { mnum[2 * i] += ln[2 * i]; mnum[2 * i + 1] += ln[2 * i + 1]; mcnt[i] += lc[i]; } }
      }
      tmsd += now() - tm;
    }
    F += n;
    for (auto& w : work) { pending.insert(pending.end(), w.seeds.begin(), w.seeds.end()); nseed_total += w.seeds.size(); w.seeds.clear(); }
    const double tc = now(); tframe += tc - tb;
    std::vector<Seed> ready, keep; keep.reserve(pending.size());
    for (const Seed& s : pending) (s.f + LMAX < F ? ready : keep).push_back(s);
    pending.swap(keep);
    if (!ready.empty()) {
#pragma omp parallel
      {
        std::vector<u64> loc((size_t)nblk * 16 * nl, 0);
#pragma omp for schedule(dynamic, 256) nowait
        for (long k = 0; k < (long)ready.size(); ++k) {
          const Seed s = ready[k]; const int cls = s.g * 4 + s.ch; u64* base = loc.data() + ((size_t)(s.f / BLK) * 16 + cls * 2) * nl;
          for (int l = 0; l < nl; ++l) {
            const int slot = (s.f + lags[l]) % NB; const unsigned short ow = owners[(size_t)slot * nh + s.h];
            if (ow == s.d) ++base[l];
            if (ow == s.d || ow == s.a) {
              const int D = ow, A = (ow == s.d) ? s.a : s.d; const double* X = pos.data() + (size_t)slot * fd;
              const double* hp = X + 3 * hs[s.h]; const double* dp = X + 3 * os[D]; const double* ap = X + 3 * os[A];
              double rx = dp[0] - ap[0], ry = dp[1] - ap[1], rz = dp[2] - ap[2]; mic(rx, ry, lx, ly); const double r2 = rx * rx + ry * ry + rz * rz;
              if (r2 <= 12.25) {
                double ux = dp[0] - hp[0], uy = dp[1] - hp[1], uz = dp[2] - hp[2]; mic(ux, uy, lx, ly); const double un = ux * ux + uy * uy + uz * uz;
                double vx = ap[0] - hp[0], vy = ap[1] - hp[1], vz = ap[2] - hp[2]; mic(vx, vy, lx, ly); const double vn = vx * vx + vy * vy + vz * vz;
                if (un > 0 && vn > 0) { const double c = (ux * vx + uy * vy + uz * vz) / std::sqrt(un * vn); if (c <= -.5 && (s.g == 0 || strict_ok(std::sqrt(r2), c))) ++base[nl + l]; }
              }
            }
          }
        }
#pragma omp critical
        for (size_t i = 0; i < loc.size(); ++i) numb[i] += loc[i];
      }
    }
    tcorr += now() - tc;
  }

  void finish(const fs::path& out, double tread_external) {
    if (Fin > F) window((int)(Fin - F));
    need(Fin == nf, "fewer frames pushed than declared");
    need(pending.empty(), "seeds left without their full lag range");
    fs::create_directories(out);
    double minOO = 1e10; std::vector<Seed> samples;
    for (auto& w : work) { for (int i = 0; i < S_NSTAT; ++i) stats[i] += w.stats[i]; for (int i = 0; i < 8; ++i) seedden[i] += w.seedden[i]; minOO = std::min(minOO, w.minoo); samples.insert(samples.end(), w.samples.begin(), w.samples.end()); }
    std::vector<u64> num((size_t)16 * nl, 0); for (int b = 0; b < nblk; ++b) for (size_t i = 0; i < (size_t)16 * nl; ++i) num[i] += numb[(size_t)b * 16 * nl + i];
    for (int g = 0; g < 2; ++g) for (int c = 0; c < 4; ++c) {
      need(num[((size_t)(g * 4 + c) * 2 + 0) * nl] == seedden[g * 4 + c], "CPT lag0 mismatch");
      need(num[((size_t)(g * 4 + c) * 2 + 1) * nl] == seedden[g * 4 + c], "CHB lag0 mismatch");
    }
    auto key = [](const Seed& a, const Seed& b) { return std::tie(a.f, a.h, a.g) < std::tie(b.f, b.h, b.g); };
    std::sort(samples.begin(), samples.end(), key);
    std::vector<int32_t> samplerows; for (auto& s : samples) for (int v : {s.f, s.h, s.d, s.a, s.g, s.ch}) samplerows.push_back(v);
    std::vector<u64> bfr(nblk);
    for (int b = 0; b < nblk; ++b) {
      bfr[b] = (u64)std::min(BLK, nf - b * BLK); oout[(size_t)b * 5 + 2] = oout[(size_t)b * 5] + oout[(size_t)b * 5 + 1];
      for (size_t i = 0; i < (size_t)360 * 360; ++i) ori[((size_t)b * 5 + 2) * 129600 + i] = ori[((size_t)b * 5) * 129600 + i] + ori[((size_t)b * 5 + 1) * 129600 + i];
    }
    writevec(out / "lag_frames.bin", lags); writevec(out / "numerators.bin", num); writevec(out / "numerators_blocks.bin", numb); writevec(out / "seed_denominators.bin", seedden);
    writevec(out / "origin_seed_denominators.bin", origin_den); writevec(out / "validation_primary_seeds.bin", samplerows);
    writevec(out / "fes_counts_halves.bin", hist); writevec(out / "fes_counts_blocks.bin", bhist); writevec(out / "validation_FES_counts.bin", testhist); writevec(out / "validation_frame_indices.bin", sampleframes);
    std::vector<int32_t> os32(os.begin(), os.end()), hs32(hs.begin(), hs.end()); writevec(out / "oxygen_coordinate_indices.bin", os32); writevec(out / "hydrogen_coordinate_indices.bin", hs32);
    writevec(out / "density.u32", dens); writevec(out / "orientation.u32", ori); writevec(out / "density_outside.u64", dout); writevec(out / "orientation_outside.u64", oout);
    writevec(out / "msd_lag_frames.i32", mlags); writevec(out / "msd_numerators.f64", mnum); writevec(out / "msd_counts.u64", mcnt);
    writevec(out / "population.u64", pop); writevec(out / "hbond_donated.u64", hbd); writevec(out / "hbond_accepted.u64", hba); writevec(out / "species.i16", tspec); writevec(out / "block_frames.u64", bfr);
    std::ofstream meta(out / "kernel_summary.json"); meta << std::setprecision(17)
      << "{\n\"kernel\":\"waterint hb_correlation cpu, threads " << nthr << "\",\n\"frames\":" << nf << ",\n\"intervals\":" << nf - 1 << ",\n\"ring_frames\":" << NB << ",\n\"block_frames\":" << BLK << ",\n\"blocks\":" << nblk
      << ",\n\"unassigned_H_total\":" << stats[S_UNASSIGNED] << ",\n\"owner_ties\":" << stats[S_OWNER_TIES] << ",\n\"primary_ties\":" << stats[S_PRIMARY_TIES]
      << ",\n\"seed_records\":" << nseed_total << ",\n\"lag_count\":" << nl << ",\n\"origin_count\":" << norig << ",\n\"elapsed_seconds\":" << now()
      << ",\n\"elapsed_read_frame_corr_seconds\":[" << tread_external << "," << tframe << "," << tcorr << "]"
      << ",\n\"msd_stride_frames\":" << MS << ",\n\"msd_ring_frames\":" << MR << ",\n\"msd_lags\":" << nml << ",\n\"msd_atoms_mobileO_H\":[" << nnl << "," << nm - nnl << "],\n\"msd_seconds\":" << tmsd
      << ",\n\"FES_exposure\":[" << stats[S_FESX0] << "," << stats[S_FESX1] << "," << stats[S_FESX2] << "," << stats[S_FESX3]
      << "],\n\"FES_outside_grid\":[" << stats[S_FESO0] << "," << stats[S_FESO1] << "," << stats[S_FESO2] << "," << stats[S_FESO3]
      << "],\n\"eligible_edges\":[" << stats[S_ELIG0] << "," << stats[S_ELIG1] << "],\n\"primary_edges\":[" << stats[S_PRIM0] << "," << stats[S_PRIM1]
      << "],\n\"region_counts\":[" << stats[S_REG0] << "," << stats[S_REG0 + 1] << "," << stats[S_REG0 + 2] << "," << stats[S_REG0 + 3] << "," << stats[S_REG0 + 4] << "," << stats[S_REG0 + 5]
      << "],\n\"species_counts\":[" << stats[S_SPC0] << "," << stats[S_SPC0 + 1] << "," << stats[S_SPC0 + 2] << "," << stats[S_SPC0 + 3] << "," << stats[S_SPC0 + 4]
      << "],\n\"minimum_eligible_OO_A\":" << minOO << "\n}\n";
    need(bool(meta), "cannot write kernel_summary.json");
  }
};
}  // namespace waterint_hbc

extern "C" {
const char* waterint_hbc_error() { return waterint_hbc::g_error.c_str(); }
void* waterint_hbc_create(int na, const int32_t* ids, const int16_t* types, const uint8_t* lattice, int64_t nframes, double lx, double ly,
                          int window, int block, int msd_stride, int msd_maxlag) {
  try { return new waterint_hbc::HBC(na, ids, types, lattice, (long)nframes, lx, ly, window, block, msd_stride, msd_maxlag); }
  catch (const std::exception& e) { waterint_hbc::g_error = e.what(); return nullptr; }
}
int waterint_hbc_push(void* h, int n, const double* zref, const double* zref2, const double* pos) {
  try { static_cast<waterint_hbc::HBC*>(h)->push(n, zref, zref2, pos); return 0; }
  catch (const std::exception& e) { waterint_hbc::g_error = e.what(); return 1; }
}
int waterint_hbc_finish(void* h, const char* outdir, double tread) {
  try { static_cast<waterint_hbc::HBC*>(h)->finish(outdir, tread); return 0; }
  catch (const std::exception& e) { waterint_hbc::g_error = e.what(); return 1; }
}
int waterint_hbc_threads(void* h) { return static_cast<waterint_hbc::HBC*>(h)->nthr; }
void waterint_hbc_free(void* h) { delete static_cast<waterint_hbc::HBC*>(h); }
}
