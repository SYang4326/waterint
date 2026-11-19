// hb_correlation_gpu.cu (waterint): CUDA streaming kernel of hb-correlation for long trajectories.
// Predicates and outputs as the CPU kernel (backends/cpu/hb_correlation.cpp); frames arrive on stdin
// (per frame: float64 z_ref, float64 z_ref2, float64 [n_OH][3]); ring buffer of W + 25000 frames; seeds from every origin f % 50 == 0 with
// f + 25000 < NFRAMES; additional per-block (BLK frames) numerators and FES histograms.
// usage: feeder | hb_correlation_gpu GEOM OUT NFRAMES LX LY [W=100000] [BLK=250000]   (GEOM: atom_ids.bin, atom_types.bin, lattice_O_ids.txt)
// Build: nvcc -O3 -std=c++17 -fmad=false -arch=sm_XX hb_correlation_gpu.cu -o hb_correlation_gpu
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
namespace fs = std::filesystem;
typedef unsigned long long u64;

static void need(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
template <class T> std::vector<T> readvec(const fs::path& p) {
  std::ifstream f(p, std::ios::binary); need(bool(f), "cannot read " + p.string());
  auto n = fs::file_size(p); need(n % sizeof(T) == 0, "bad vector size"); std::vector<T> v(n / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), n); need(bool(f), "truncated vector"); return v;
}
template <class T> void writevec(const fs::path& p, const std::vector<T>& v) {
  std::ofstream f(p, std::ios::binary); need(bool(f), "cannot write " + p.string());
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T)); need(bool(f), "write failed");
}

struct Seed { int32_t f, h, d, a, g, ch; };
// statistics slots (u64)
enum { S_UNASSIGNED, S_OWNER_TIES, S_PRIMARY_TIES, S_ELIG0, S_ELIG1, S_PRIM0, S_PRIM1, S_FESX0, S_FESX1, S_FESX2, S_FESX3,
       S_FESO0, S_FESO1, S_FESO2, S_FESO3, S_REG0, S_SPC0 = S_REG0 + 6, S_TS0 = S_SPC0 + 5, S_NSTAT = S_TS0 + 4 };   // S_TS: per-frame Os-H, L1 OH-, L2 OH-, bulk OH-

struct Params {
  int nf, na, no, nh, norig, nl, nsample; double lx, ly;
  int nb, f0, blk; unsigned int* bhist;
  unsigned int* dens; unsigned int* ori; u64* dout; u64* oout; u64* pop; u64* hbd; u64* hba; short* tspec; const double* zref2;   // structure
  int ms, mr, nm, nnl; const int* nlo; unsigned char* mcode;   // MSD: stride, ring frames, atoms, mobile O; O -> mobile ordinal; origin codes   // stream: ring size, first global frame of this launch, frames per block
  const double* pos; const double* zref; const int* os; const int* hs; const int* oid; const unsigned char* lattice;
  const int* lags; const int* sampleframes;
  unsigned short* owners; u64* stats; u64* minoo_bits; Seed* seeds; u64* nseeds; u64 seedcap; Seed* samples; u64* nsamples;
  u64* seedden; u64* origin_den; u64* hist; u64* testhist;
};

__device__ __forceinline__ void mic(double& x, double& y, double lx, double ly) { x -= lx * rint(x / lx); y -= ly * rint(y / ly); }
__device__ __forceinline__ int direction(int rd, int cd, int ra, int ca) {
  if (rd == 1 && cd == 1 && ra == 1 && ca == 2) return 0;
  if (rd == 1 && cd == 2 && ra == 2 && ca == 1) return 1;
  if (rd == 1 && cd == 2 && ra == 1 && ca == 1) return 2;
  if (rd == 2 && cd == 2 && ra == 1 && ca == 1) return 3;
  return -1;
}
__device__ __forceinline__ int binof(double x, double lo, double step, int n) { int b = (int)floor((x - lo) / step); return b >= 0 && b < n ? b : -1; }
// conventions: inclusive upper edge; O channel = lattice ? kind : 5 + region*5 + kind (kind = min(nH, 4))
__device__ __forceinline__ int binc(double x, double lo, double hi, int n) { if (x < lo || x > hi) return -1; if (x == hi) return n - 1; return (int)floor((x - lo) / (hi - lo) * n); }
__device__ __forceinline__ int och(int lat, int reg, int cnt) { const int kind = min(cnt, 4); if (lat) return kind; const int r = reg == 1 ? 0 : (reg == 2 ? 1 : (reg == 3 ? 2 : 3)); return 5 + r * 5 + kind; }
__device__ __forceinline__ bool strict_ok(double r, double c) { return r >= 2.2 && r < 3.2 && c <= -0.86602540378443864676; }

// one block per frame, one thread per H
__global__ void __launch_bounds__(1024) frame_kernel(Params P) {
  extern __shared__ double sh[];
  double* X = sh;                                   // [na*3]
  int* cnt = (int*)(X + 3 * P.na);                  // [no]
  int* reg = cnt + P.no;                            // [no]
  u64* bstat = (u64*)(reg + P.no + (P.no & 1));     // [S_NSTAT] block partial sums (8-byte aligned)
  int* acc = (int*)(bstat + S_NSTAT);               // [2*no] accepted eligible H bonds per O (broad, strict)
  int* reg2 = acc + 2 * P.no;                       // [no] region of each O from the second reference plane
  const int f = P.f0 + blockIdx.x; const int t = threadIdx.x;      // f = global frame index
  const int slot = f % P.nb;                                       // ring-buffer slot
  const double* src = P.pos + (size_t)slot * P.na * 3;
  for (int i = t; i < 3 * P.na; i += blockDim.x) X[i] = src[i];
  for (int i = t; i < P.no; i += blockDim.x) cnt[i] = 0;
  for (int i = t; i < 2 * P.no; i += blockDim.x) acc[i] = 0;
  for (int i = t; i < S_NSTAT; i += blockDim.x) bstat[i] = 0;
  __syncthreads();
  const double lx = P.lx, ly = P.ly;
  // ---- owner of H t
  int owner = -1;
  if (t < P.nh) {
    const double* hp = X + 3 * P.hs[t];
    double best = 1e100;
    for (int o = 0; o < P.no; ++o) {
      const double* q = X + 3 * P.os[o]; double dx = hp[0] - q[0], dy = hp[1] - q[1], dz = hp[2] - q[2]; mic(dx, dy, lx, ly);
      double dd = dx * dx + dy * dy + dz * dz; if (dd <= 2.25 && dd < best) best = dd;
    }
    double lim = pow(sqrt(best) + 1e-8, 2.0); int ntie = 0;
    if (best < 1e99) {
      for (int o = 0; o < P.no; ++o) {
        const double* q = X + 3 * P.os[o]; double dx = hp[0] - q[0], dy = hp[1] - q[1], dz = hp[2] - q[2]; mic(dx, dy, lx, ly);
        double dd = dx * dx + dy * dy + dz * dz;
        if (dd <= 2.25 && dd <= lim) { ++ntie; if (owner < 0 || P.oid[P.os[o]] < P.oid[P.os[owner]]) owner = o; }
      }
    }
    if (ntie > 1) atomicAdd(&bstat[S_OWNER_TIES], 1ULL);
    if (owner < 0) atomicAdd(&bstat[S_UNASSIGNED], 1ULL); else atomicAdd(&cnt[owner], 1);
    P.owners[(size_t)slot * P.nh + t] = owner < 0 ? 65535 : (unsigned short)owner;
  }
  __syncthreads();
  // ---- regions and species counts
  const double zr = P.zref[slot];
  for (int o = t; o < P.no; o += blockDim.x) {
    double z = X[3 * P.os[o] + 2] - zr;
    int r = P.lattice[o] ? 0 : (z >= 1.5 && z < 2.8 ? 1 : (z >= 2.8 && z < 4.0 ? 2 : (z >= 4 && z < 30 ? 3 : (z < 1.5 ? 4 : 5))));
    reg[o] = r; atomicAdd(&bstat[S_REG0 + r], 1ULL); atomicAdd(&bstat[S_SPC0 + min(cnt[o], 4)], 1ULL);
    {                                               // density (O-centred species), population, per-frame OH counts
      // second reference plane (sequential mean of the Mg within 0.70 A of the highest Mg): own z and region
      const double z2 = X[3 * P.os[o] + 2] - P.zref2[slot];
      const int r2 = P.lattice[o] ? 0 : (z2 >= 1.5 && z2 < 2.8 ? 1 : (z2 >= 2.8 && z2 < 4.0 ? 2 : (z2 >= 4 && z2 < 30 ? 3 : (z2 < 1.5 ? 4 : 5))));
      reg2[o] = r2;
      if (f % P.ms == 0 && P.nlo[o] >= 0) P.mcode[(size_t)((f / P.ms) % P.mr) * P.nm + P.nlo[o]] = (unsigned char)r2;
      const int chd = och(P.lattice[o], r2, cnt[o]); const size_t bc = (size_t)(f / P.blk) * 25 + chd;
      atomicAdd(&P.pop[bc], 1ULL); const int iz = binc(z2, -1.0, 30.0, 1240);
      if (iz >= 0) atomicAdd(&P.dens[bc * 1240 + iz], 1u); else atomicAdd(&P.dout[bc], 1ULL);
      if (cnt[o] == 1) { if (P.lattice[o]) atomicAdd(&bstat[S_TS0], 1ULL); else if (r2 >= 1 && r2 <= 3) atomicAdd(&bstat[S_TS0 + r2], 1ULL); }
    }
  }
  __syncthreads();
  // ---- eligible edges, primary acceptors, seeds, FES
  if (t < P.nh && f % P.ms == 0)                    // H: region of its owner (0 lattice = Os-H, 1 L1, 2 L2, 3 bulk, 4/5 other, 6 free)
    P.mcode[(size_t)((f / P.ms) % P.mr) * P.nm + P.nnl + t] = (unsigned char)(owner < 0 ? 6 : (P.lattice[owner] ? 0 : reg2[owner]));
  const bool origin = (f % 50 == 0) && (f + 25000 < P.nf);
  bool sample = false; for (int i = 0; i < P.nsample; ++i) sample |= (P.sampleframes[i] == f);
  double minoo = 1e10;
  if (t < P.nh && owner >= 0) {
    const int d = owner; const double* hp = X + 3 * P.hs[t]; const double* dp = X + 3 * P.os[d];
    double ux = dp[0] - hp[0], uy = dp[1] - hp[1], uz = dp[2] - hp[2]; mic(ux, uy, lx, ly); double un = ux * ux + uy * uy + uz * uz;
    {                                               // O-H orientation: O -> H vs +z, z of the O
      int cho = -1; if (P.lattice[d]) { if (cnt[d] == 1) cho = 1; } else { if (cnt[d] == 1) cho = 0; else if (cnt[d] == 2) cho = 3; else if (cnt[d] == 3) cho = 4; }
      if (cho >= 0 && un > 0) {
        const double a = acos(fmax(-1.0, fmin(1.0, (-uz) / sqrt(un)))) * 180 / 3.14159265358979323846;
        const int iz = binc(dp[2] - P.zref2[slot], -1.0, 8.0, 360), ia = binc(a, 0.0, 180.0, 360); const size_t bc = (size_t)(f / P.blk) * 5 + cho;
        if (iz >= 0 && ia >= 0) atomicAdd(&P.ori[(bc * 360 + iz) * 360 + ia], 1u); else atomicAdd(&P.oout[bc], 1ULL);
      }
    }
    double best[2] = {1e100, 1e100}; u64 el0 = 0, el1 = 0;
    for (int a = 0; a < P.no; ++a) {
      if (a == d) continue; const double* ap = X + 3 * P.os[a];
      double rx = dp[0] - ap[0], ry = dp[1] - ap[1], rz = dp[2] - ap[2]; mic(rx, ry, lx, ly); double r2 = rx * rx + ry * ry + rz * rz;
      if (r2 > 12.25) continue;
      double vx = ap[0] - hp[0], vy = ap[1] - hp[1], vz = ap[2] - hp[2]; mic(vx, vy, lx, ly); double vn = vx * vx + vy * vy + vz * vz;
      if (un <= 0 || vn <= 0) continue;
      double c = (ux * vx + uy * vy + uz * vz) / sqrt(un * vn); if (c > -.5) continue;
      double r = sqrt(r2), ha = sqrt(vn); minoo = fmin(minoo, r); ++el0; atomicAdd(&acc[a], 1);
      bool st = strict_ok(r, c); if (st) { ++el1; atomicAdd(&acc[P.no + a], 1); }
      best[0] = fmin(best[0], ha); if (st) best[1] = fmin(best[1], ha);
    }
    atomicAdd(&bstat[S_ELIG0], el0); atomicAdd(&bstat[S_ELIG1], el1);
    {                                               // donated eligible H bonds of this H, by donor channel (0, 1, 2, >= 3)
      const size_t bc = ((size_t)(f / P.blk) * 2) * 25 + och(P.lattice[d], reg[d], cnt[d]);
      atomicAdd(&P.hbd[bc * 4 + min((int)el0, 3)], 1ULL); atomicAdd(&P.hbd[(bc + 25) * 4 + min((int)el1, 3)], 1ULL);
    }
    for (int g = 0; g < 2; ++g) {
      int chosen = -1, ties = 0; double cr = 0, cha = 0;
      for (int a = 0; a < P.no; ++a) {
        if (a == d) continue; const double* ap = X + 3 * P.os[a];
        double rx = dp[0] - ap[0], ry = dp[1] - ap[1], rz = dp[2] - ap[2]; mic(rx, ry, lx, ly); double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 > 12.25) continue;
        double vx = ap[0] - hp[0], vy = ap[1] - hp[1], vz = ap[2] - hp[2]; mic(vx, vy, lx, ly); double vn = vx * vx + vy * vy + vz * vz;
        if (un <= 0 || vn <= 0) continue;
        double c = (ux * vx + uy * vy + uz * vz) / sqrt(un * vn); if (c > -.5) continue;
        double r = sqrt(r2), ha = sqrt(vn);
        if (g == 1 && !strict_ok(r, c)) continue;
        if (ha <= best[g] + 1e-8) { ++ties; if (chosen < 0 || P.oid[P.os[a]] < P.oid[P.os[chosen]]) { chosen = a; cr = r; cha = ha; } }
      }
      if (ties > 1) atomicAdd(&bstat[S_PRIMARY_TIES], 1ULL);
      if (chosen < 0) continue;
      atomicAdd(&bstat[g ? S_PRIM1 : S_PRIM0], 1ULL);
      const int a = chosen; const int ch = direction(reg[d], cnt[d], reg[a], cnt[a]);
      if (origin && ch >= 0) {
        u64 k = atomicAdd(P.nseeds, 1ULL);
        if (k < P.seedcap) P.seeds[k] = Seed{f, t, d, a, g, ch};
        atomicAdd(&P.seedden[g * 4 + ch], 1ULL); atomicAdd(&P.origin_den[(size_t)(g * 4 + ch) * P.norig + f / 50], 1ULL);
      }
      if (sample && ch >= 0) { u64 k = atomicAdd(P.nsamples, 1ULL); P.samples[k] = Seed{f, t, d, a, g, ch}; }
      if (cnt[d] != 2 || cnt[a] != 1 || P.lattice[d] || P.lattice[a]) continue;
      int fc = -1, oa = -1, ob = -1;
      if (reg[d] == 1 && reg[a] == 1) { fc = 0; oa = P.oid[P.os[d]] < P.oid[P.os[a]] ? d : a; ob = oa == d ? a : d; }
      else if ((reg[d] == 1 && reg[a] == 2) || (reg[d] == 2 && reg[a] == 1)) { fc = 1; oa = reg[d] == 1 ? d : a; ob = oa == d ? a : d; }
      if (fc < 0) continue;
      atomicAdd(&bstat[S_FESX0 + g * 2 + fc], 1ULL);
      const double* pa = X + 3 * P.os[oa]; const double* pb = X + 3 * P.os[ob];
      double ax = pa[0] - hp[0], ay = pa[1] - hp[1], az = pa[2] - hp[2]; mic(ax, ay, lx, ly);
      double bx = pb[0] - hp[0], by = pb[1] - hp[1], bz = pb[2] - hp[2]; mic(bx, by, lx, ly);
      double delta = sqrt(ax * ax + ay * ay + az * az) - sqrt(bx * bx + by * by + bz * bz);
      int db = binof(delta, -3.5, .025, 280), rb = binof(cr, 0, .01, 350);
      if (db < 0 || rb < 0) { atomicAdd(&bstat[S_FESO0 + g * 2 + fc], 1ULL); continue; }
      int half = f < (P.nf - 1) / 2 ? 0 : 1;
      atomicAdd(&P.hist[((((size_t)g * 2 + fc) * 2 + half) * 280 + db) * 350 + rb], 1ULL);
      atomicAdd(&P.bhist[((((size_t)(f / P.blk)) * 4 + g * 2 + fc) * 280 + db) * 350 + rb], 1u);
      if (sample) atomicAdd(&P.testhist[(((size_t)g * 2 + fc) * 280 + db) * 350 + rb], 1ULL);
      (void)cha;
    }
  }
  // minimum eligible O-O (non-negative doubles: bit patterns order like the values)
  if (minoo < 1e10) atomicMin(P.minoo_bits, (u64)__double_as_longlong(minoo));
  __syncthreads();
  for (int o = t; o < P.no; o += blockDim.x) {      // accepted eligible H bonds per O (0 ... >= 7), by O channel
    const size_t bc = ((size_t)(f / P.blk) * 2) * 25 + och(P.lattice[o], reg[o], cnt[o]);
    atomicAdd(&P.hba[bc * 8 + min(acc[o], 7)], 1ULL); atomicAdd(&P.hba[(bc + 25) * 8 + min(acc[P.no + o], 7)], 1ULL);
  }
  if (t < 4) P.tspec[(size_t)f * 4 + t] = (short)bstat[S_TS0 + t];
  for (int i = t; i < S_NSTAT; i += blockDim.x) if (bstat[i]) atomicAdd(&P.stats[i], bstat[i]);
}

// correlations: thread per (lag, chunk of seeds); seeds sorted by frame; numerators per time block
__global__ void corr_kernel(const Seed* seeds, u64 nseeds, int chunk, const int* lags, int nl, const double* pos, const unsigned short* owners,
                            int na, int nh, int nb, int blk, const int* os, const int* hs, double lx, double ly, u64* numer) {
  const int l = blockIdx.y; const u64 c0 = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * chunk;
  if (l >= nl || c0 >= nseeds) return;
  u64 cnt[16] = {0}; int cur = -1;
  const u64 c1 = min(nseeds, c0 + chunk); const int lag = lags[l];
  for (u64 k = c0; k <= c1; ++k) {
    const int b = k < c1 ? seeds[k].f / blk : -2;
    if (b != cur) {                                   // flush the finished block
      if (cur >= 0) for (int i = 0; i < 16; ++i) if (cnt[i]) { atomicAdd(&numer[((size_t)cur * 16 + i) * nl + l], cnt[i]); cnt[i] = 0; }
      cur = b; if (k == c1) break;
    }
    const Seed s = seeds[k]; const int slot = (s.f + lag) % nb;
    const unsigned short ow = owners[(size_t)slot * nh + s.h];
    const int cls = s.g * 4 + s.ch;
    if (ow == s.d) ++cnt[cls * 2 + 0];
    if (ow == s.d || ow == s.a) {
      const int D = ow, A = (ow == s.d) ? s.a : s.d;
      const double* X = pos + (size_t)slot * na * 3;
      const double* hp = X + 3 * hs[s.h]; const double* dp = X + 3 * os[D]; const double* ap = X + 3 * os[A];
      double rx = dp[0] - ap[0], ry = dp[1] - ap[1], rz = dp[2] - ap[2]; mic(rx, ry, lx, ly); double r2 = rx * rx + ry * ry + rz * rz;
      if (r2 <= 12.25) {
        double ux = dp[0] - hp[0], uy = dp[1] - hp[1], uz = dp[2] - hp[2]; mic(ux, uy, lx, ly); double un = ux * ux + uy * uy + uz * uz;
        double vx = ap[0] - hp[0], vy = ap[1] - hp[1], vz = ap[2] - hp[2]; mic(vx, vy, lx, ly); double vn = vx * vx + vy * vy + vz * vz;
        if (un > 0 && vn > 0) {
          double c = (ux * vx + uy * vy + uz * vz) / sqrt(un * vn);
          if (c <= -.5 && (s.g == 0 || strict_ok(sqrt(r2), c))) ++cnt[cls * 2 + 1];
        }
      }
    }
  }
}

// xy MSD at sparse lags, every strided frame as origin (estimator N(k) = sum_i,t m(i,t) [dx^2 + dy^2], D(k) = sum m):
// one block per (later strided frame j, lag index l); classes = species (0 mobile O, 1 H) x origin code (0..6), by the block of the origin
__global__ void msd_kernel(int j0, int nml, const int* mlags, const double* mu, const unsigned char* mcode, int mr, int nm, int nnl,
                           int ms, int blk, double* mnum, u64* mcnt) {
  const int j = j0 + blockIdx.x, l = blockIdx.y; const int jo = j - mlags[l]; if (jo < 0) return;
  const double* a = mu + (size_t)(j % mr) * nm * 2; const double* b = mu + (size_t)(jo % mr) * nm * 2; const unsigned char* code = mcode + (size_t)(jo % mr) * nm;
  double sx[14], sy[14]; unsigned int c[14]; for (int k = 0; k < 14; ++k) { sx[k] = 0; sy[k] = 0; c[k] = 0; }
  for (int i = threadIdx.x; i < nm; i += blockDim.x) {
    const int cls = (i < nnl ? 0 : 7) + code[i]; const double dx = a[2 * i] - b[2 * i], dy = a[2 * i + 1] - b[2 * i + 1];
    sx[cls] += dx * dx; sy[cls] += dy * dy; ++c[cls];
  }
  const size_t base = ((size_t)((long)jo * ms / blk) * 14) * nml;
  for (int k = 0; k < 14; ++k) if (c[k]) {
    const size_t q = base + (size_t)k * nml + l; atomicAdd(&mnum[2 * q], sx[k]); atomicAdd(&mnum[2 * q + 1], sy[k]); atomicAdd(&mcnt[q], (u64)c[k]);
  }
}

static bool readn(void* p, size_t n) { return std::fread(p, 1, n, stdin) == n; }

int main(int argc, char** argv) {
  try {
    need(argc >= 6 && argc <= 10, "usage: feeder | hb_correlation_gpu GEOM OUT NFRAMES LX LY [W=100000] [BLK=250000] [MSD_STRIDE=10] [MSD_MAXLAG=250000 strided frames]");
    const int MS = argc > 8 ? std::stoi(argv[8]) : 10; const int MAXLAG = argc > 9 ? std::stoi(argv[9]) : 250000;
    fs::path geom = argv[1], out = argv[2]; const long nftot = std::stol(argv[3]); double lx = std::stod(argv[4]), ly = std::stod(argv[5]);
    const int W = argc > 6 ? std::stoi(argv[6]) : 100000, BLK = argc > 7 ? std::stoi(argv[7]) : 250000;
    const int LMAX = 25000; need(nftot > LMAX && nftot < 2000000000L && W >= 50 && BLK >= 50, "bad NFRAMES / W / BLK"); fs::create_directories(out);
    const int nf = (int)nftot, NB = W + LMAX, nblk = (nf - 1) / BLK + 1;
    std::setvbuf(stdin, nullptr, _IOFBF, 64 << 20);
    { int c0 = std::fgetc(stdin); need(c0 != EOF, "empty frame stream"); std::ungetc(c0, stdin); }   // the feeder writes GEOM before its first record
    auto ids = readvec<int32_t>(geom / "atom_ids.bin"); auto types = readvec<int16_t>(geom / "atom_types.bin");
    const int na = (int)ids.size(); need((int)types.size() == na, "bad geometry metadata");
    std::vector<int> os, hs; for (int i = 0; i < na; ++i) { if (types[i] == 3) os.push_back(i); else if (types[i] == 1) hs.push_back(i); else need(false, "unexpected selected atom type"); }
    const int no = os.size(), nh = hs.size(); need(nh <= 1024, "too many H for one block");
    std::ifstream latf(geom / "lattice_O_ids.txt"); need(bool(latf), "missing lattice IDs"); std::vector<int> latids; int lid; while (latf >> lid) latids.push_back(lid);
    std::vector<unsigned char> lattice(no); for (int o = 0; o < no; ++o) lattice[o] = std::find(latids.begin(), latids.end(), ids[os[o]]) != latids.end();
    need((size_t)std::count(lattice.begin(), lattice.end(), 1) == latids.size(), "lattice mapping failure");
    std::vector<int32_t> lags; for (int i = 0; i <= 250; ++i) lags.push_back(i); for (int i = 260; i <= 2500; i += 10) lags.push_back(i); for (int i = 2550; i <= 25000; i += 50) lags.push_back(i);
    const int nl = lags.size(), norig = (nf - 25001) / 50 + 1;
    std::vector<int> sampleframes = {0, 1, 50, 250, 2500, 12500, 25000, nf / 2, nf - 25001, nf - 1};
    std::sort(sampleframes.begin(), sampleframes.end()); sampleframes.erase(std::unique(sampleframes.begin(), sampleframes.end()), sampleframes.end());
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));   // do not spin on the CPU while the GPU works: the job has one core, shared with the feeder
    auto t0 = std::chrono::steady_clock::now(); auto now = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    // ---- device buffers
    const size_t fdoubles = (size_t)na * 3, fbytes = fdoubles * sizeof(double);
    double* dpos; CK(cudaMalloc(&dpos, fbytes * NB)); double* dzref; CK(cudaMalloc(&dzref, sizeof(double) * NB)); double* dzref2; CK(cudaMalloc(&dzref2, sizeof(double) * NB));
    auto up = [](const void* h, size_t n) { void* d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); return d; };
    std::vector<int> oid(ids.begin(), ids.end());
    Params P{}; P.nf = nf; P.na = na; P.no = no; P.nh = nh; P.norig = norig; P.nl = nl; P.nsample = sampleframes.size(); P.lx = lx; P.ly = ly;
    P.nb = NB; P.blk = BLK; P.pos = dpos; P.zref = dzref; P.zref2 = dzref2;
    P.os = (int*)up(os.data(), sizeof(int) * no); P.hs = (int*)up(hs.data(), sizeof(int) * nh);
    P.oid = (int*)up(oid.data(), sizeof(int) * na); P.lattice = (unsigned char*)up(lattice.data(), no); P.lags = (int*)up(lags.data(), sizeof(int) * nl);
    P.sampleframes = (int*)up(sampleframes.data(), sizeof(int) * sampleframes.size());
    CK(cudaMalloc(&P.owners, sizeof(unsigned short) * (size_t)NB * nh));
    auto zalloc = [](u64** p, size_t n) { CK(cudaMalloc(p, sizeof(u64) * n)); CK(cudaMemset(*p, 0, sizeof(u64) * n)); };
    zalloc(&P.stats, S_NSTAT); zalloc(&P.nseeds, 1); zalloc(&P.nsamples, 1); zalloc(&P.seedden, 8); zalloc(&P.origin_den, 8 * (size_t)norig);
    zalloc(&P.hist, (size_t)2 * 2 * 2 * 280 * 350); zalloc(&P.testhist, (size_t)2 * 2 * 280 * 350);
    const size_t nbh = (size_t)nblk * 4 * 280 * 350; CK(cudaMalloc(&P.bhist, sizeof(unsigned int) * nbh)); CK(cudaMemset(P.bhist, 0, sizeof(unsigned int) * nbh));
    u64 big = (u64)__builtin_bit_cast(u64, 1e10); CK(cudaMalloc(&P.minoo_bits, sizeof(u64))); CK(cudaMemcpy(P.minoo_bits, &big, sizeof(u64), cudaMemcpyHostToDevice));
    P.seedcap = (u64)(W / 50 + 2) * nh * 2; CK(cudaMalloc(&P.seeds, sizeof(Seed) * P.seedcap));
    CK(cudaMalloc(&P.samples, sizeof(Seed) * (size_t)sampleframes.size() * nh * 2));
    const u64 readycap = (u64)(NB / 50 + 4) * nh * 2; Seed* dready; CK(cudaMalloc(&dready, sizeof(Seed) * readycap));
    u64* numer; zalloc(&numer, (size_t)nblk * 16 * nl);
    // ---- MSD: mobile (non-lattice) O then H; unwrapped xy of the strided frames in a ring; sparse lags in strided frames
    need(MS >= 1 && MAXLAG >= 1, "bad MSD stride / max lag"); const int nsf = (nf - 1) / MS + 1;      // strided frames 0, MS, 2 MS, ...
    std::vector<int> nlo(no, -1), matom; for (int o = 0; o < no; ++o) if (!lattice[o]) { nlo[o] = (int)matom.size(); matom.push_back(os[o]); }
    const int nnl = (int)matom.size(); for (int t = 0; t < nh; ++t) matom.push_back(hs[t]); const int nm = (int)matom.size();
    std::vector<int32_t> mlags; { const int top = std::min(MAXLAG, nsf - 1);
      for (int i = 0; i <= 100 && i <= top; ++i) mlags.push_back(i); for (int i = 110; i <= 1000 && i <= top; i += 10) mlags.push_back(i);
      for (int i = 1100; i <= 10000 && i <= top; i += 100) mlags.push_back(i); for (int i = 11000; i <= top; i += 1000) mlags.push_back(i); }
    const int nml = (int)mlags.size(), MR = std::min(nsf, mlags.back() + W / MS + 2);
    double* dmu; CK(cudaMalloc(&dmu, sizeof(double) * (size_t)MR * nm * 2)); CK(cudaMalloc(&P.mcode, (size_t)MR * nm));
    P.ms = MS; P.mr = MR; P.nm = nm; P.nnl = nnl; P.nlo = (int*)up(nlo.data(), sizeof(int) * no);
    int* dmlags = (int*)up(mlags.data(), sizeof(int) * nml);
    const size_t nmq = (size_t)nblk * 14 * nml; double* dmnum; CK(cudaMalloc(&dmnum, sizeof(double) * 2 * nmq)); CK(cudaMemset(dmnum, 0, sizeof(double) * 2 * nmq)); u64* dmcnt; zalloc(&dmcnt, nmq);
    std::vector<double> mprev((size_t)nm * 2, 0.0), mimg((size_t)nm * 2, 0.0), mhost; const double mL[2] = {lx, ly}; double tmsd = 0;
    const size_t nde = (size_t)nblk * 25 * 1240, nor = (size_t)nblk * 5 * 360 * 360;
    CK(cudaMalloc(&P.dens, sizeof(unsigned int) * nde)); CK(cudaMemset(P.dens, 0, sizeof(unsigned int) * nde));
    CK(cudaMalloc(&P.ori, sizeof(unsigned int) * nor)); CK(cudaMemset(P.ori, 0, sizeof(unsigned int) * nor));
    zalloc(&P.dout, (size_t)nblk * 25); zalloc(&P.oout, (size_t)nblk * 5); zalloc(&P.pop, (size_t)nblk * 25);
    zalloc(&P.hbd, (size_t)nblk * 2 * 25 * 4); zalloc(&P.hba, (size_t)nblk * 2 * 25 * 8);
    CK(cudaMalloc(&P.tspec, sizeof(short) * (size_t)nf * 4));
    int threads = ((nh + 31) / 32) * 32;
    size_t shm = sizeof(double) * 3 * na + sizeof(int) * (2 * no + (no & 1)) + sizeof(u64) * S_NSTAT + sizeof(int) * 3 * no;
    CK(cudaFuncSetAttribute(frame_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    // ---- stream
    const int sub = 2000; double* hbuf; CK(cudaMallocHost(&hbuf, (fbytes + sizeof(double)) * sub)); std::vector<double> hz(sub), hz2(sub);
    // reader thread: stdin -> bounded queue of blocks (<= sub frames each), so the feeder keeps decoding during the GPU work
    const size_t recd = fdoubles + 2, qmax = 40; std::deque<std::vector<double>> queue; std::mutex qm; std::condition_variable qcv; std::string rerr; bool rdone = false;
    std::thread reader([&]() {
      long left = nf;
      while (left > 0) {
        const int m = (int)std::min<long>(sub, left); std::vector<double> blockbuf((size_t)m * recd);
        if (!readn(blockbuf.data(), sizeof(double) * blockbuf.size())) { std::lock_guard<std::mutex> g(qm); rerr = "short frame stream"; rdone = true; qcv.notify_all(); return; }
        bool finite = true; for (double v : blockbuf) finite &= std::isfinite(v);
        std::unique_lock<std::mutex> g(qm);
        if (!finite) { rerr = "nonfinite coordinate"; rdone = true; qcv.notify_all(); return; }
        qcv.wait(g, [&]() { return queue.size() < qmax; }); queue.push_back(std::move(blockbuf)); left -= m; qcv.notify_all();
      }
      std::lock_guard<std::mutex> g(qm); rdone = true; qcv.notify_all();
    });
    struct TG { std::thread& t; ~TG() { if (t.joinable()) t.detach(); } } tguard{reader};   // error paths: do not terminate on a joinable thread
    std::vector<double> cur; size_t curpos = 0;        // current block and the next unread frame in it
    auto nextframe = [&]() -> const double* {
      if (curpos * recd >= cur.size()) {
        std::unique_lock<std::mutex> g(qm); qcv.wait(g, [&]() { return !queue.empty() || rdone; });
        if (queue.empty()) { g.unlock(); if (reader.joinable()) reader.join(); throw std::runtime_error(rerr.empty() ? "frame stream ended early" : rerr); }
        cur = std::move(queue.front()); queue.pop_front(); curpos = 0; qcv.notify_all();
      }
      return cur.data() + (curpos++) * recd;
    };
    std::vector<Seed> pending, ready, fresh; u64 nseed_total = 0; double tread = 0, tframe = 0, tcorr = 0;
    auto key = [](const Seed& a, const Seed& b) { return std::tie(a.f, a.h, a.g) < std::tie(b.f, b.h, b.g); };
    for (int F = 0; F < nf;) {
      const int n = std::min(W, nf - F); double ta = now();
      for (int g0 = 0; g0 < n;) {                     // upload in pieces that do not cross the ring boundary
        const int slot = (F + g0) % NB; const int m = std::min(std::min(sub, n - g0), NB - slot);
        for (int i = 0; i < m; ++i) {
          const double* rec = nextframe();
          hz[i] = rec[0]; hz2[i] = rec[1]; std::memcpy(hbuf + (size_t)i * fdoubles, rec + 2, fbytes);
          const int fg = F + g0 + i;
          if (fg % MS == 0) {                           // unwrap x, y against the previous strided frame (steps << L/2)
            const size_t off = mhost.size(); mhost.resize(off + (size_t)nm * 2); const double* X = rec + 2;
            for (int a = 0; a < nm; ++a) for (int ax = 0; ax < 2; ++ax) {
              const double x = X[3 * matom[a] + ax]; const size_t q = (size_t)a * 2 + ax;
              if (fg > 0) mimg[q] -= std::rint((x - mprev[q]) / mL[ax]);
              mprev[q] = x; mhost[off + q] = x + mL[ax] * mimg[q];
            }
          }
        }
        CK(cudaMemcpy(dpos + (size_t)slot * fdoubles, hbuf, fbytes * m, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dzref + slot, hz.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dzref2 + slot, hz2.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        g0 += m;
      }
      double tb = now(); tread += tb - ta;
      CK(cudaMemset(P.nseeds, 0, sizeof(u64))); P.f0 = F;
      frame_kernel<<<n, threads, shm>>>(P); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      {                                                 // MSD for the strided frames of this window
        double tm = now(); const int j0 = (F + MS - 1) / MS, nj = (int)(mhost.size() / ((size_t)nm * 2));
        for (int g = 0; g < nj;) { const int slot = (j0 + g) % MR, m = std::min(nj - g, MR - slot);
          CK(cudaMemcpy(dmu + (size_t)slot * nm * 2, mhost.data() + (size_t)g * nm * 2, sizeof(double) * (size_t)m * nm * 2, cudaMemcpyHostToDevice)); g += m; }
        if (nj > 0) { dim3 mg(nj, nml); msd_kernel<<<mg, 64>>>(j0, nml, dmlags, dmu, P.mcode, MR, nm, nnl, MS, BLK, dmnum, dmcnt); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        mhost.clear(); tmsd += now() - tm;
      }
      F += n;
      u64 ns; CK(cudaMemcpy(&ns, P.nseeds, sizeof(u64), cudaMemcpyDeviceToHost)); need(ns <= P.seedcap, "seed buffer overflow");
      fresh.resize(ns); if (ns) CK(cudaMemcpy(fresh.data(), P.seeds, sizeof(Seed) * ns, cudaMemcpyDeviceToHost));
      pending.insert(pending.end(), fresh.begin(), fresh.end()); nseed_total += ns;
      double tc = now(); tframe += tc - tb;
      ready.clear(); std::vector<Seed> keep; keep.reserve(pending.size());
      for (const Seed& s : pending) (s.f + LMAX < F ? ready : keep).push_back(s);
      pending.swap(keep);
      if (!ready.empty()) {
        need(ready.size() <= readycap, "ready seed buffer overflow"); std::sort(ready.begin(), ready.end(), key);
        CK(cudaMemcpy(dready, ready.data(), sizeof(Seed) * ready.size(), cudaMemcpyHostToDevice));
        const int chunk = 256, tpb = 128; u64 nthr = (ready.size() + chunk - 1) / chunk; dim3 grid((unsigned)((nthr + tpb - 1) / tpb), nl);
        corr_kernel<<<grid, tpb>>>(dready, ready.size(), chunk, P.lags, nl, dpos, P.owners, na, nh, NB, BLK, P.os, P.hs, lx, ly, numer);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      }
      tcorr += now() - tc;
      std::cout << "frames " << F << "/" << nf << " seeds " << nseed_total << " pending " << pending.size() << " elapsed " << now() << std::endl;
    }
    need(pending.empty(), "seeds left without their full lag range"); CK(cudaFreeHost(hbuf)); if (reader.joinable()) reader.join();
    // ---- back to host
    auto down = [](const void* d, size_t n, void* h) { CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); };
    std::vector<u64> stats(S_NSTAT), seedden(8), origin_den(8 * (size_t)norig), hist((size_t)2 * 2 * 2 * 280 * 350), testhist((size_t)2 * 2 * 280 * 350), numb((size_t)nblk * 16 * nl), num((size_t)16 * nl, 0);
    std::vector<unsigned int> bhist(nbh);
    down(P.stats, sizeof(u64) * S_NSTAT, stats.data()); down(P.seedden, sizeof(u64) * 8, seedden.data()); down(P.origin_den, sizeof(u64) * origin_den.size(), origin_den.data());
    down(P.hist, sizeof(u64) * hist.size(), hist.data()); down(P.testhist, sizeof(u64) * testhist.size(), testhist.data()); down(numer, sizeof(u64) * numb.size(), numb.data());
    down(P.bhist, sizeof(unsigned int) * nbh, bhist.data());
    for (int b = 0; b < nblk; ++b) for (size_t i = 0; i < (size_t)16 * nl; ++i) num[i] += numb[(size_t)b * 16 * nl + i];
    for (int g = 0; g < 2; ++g) for (int c = 0; c < 4; ++c) {
      need(num[((size_t)(g * 4 + c) * 2 + 0) * nl] == seedden[g * 4 + c], "CPT lag0 mismatch");
      need(num[((size_t)(g * 4 + c) * 2 + 1) * nl] == seedden[g * 4 + c], "CHB lag0 mismatch");
    }
    u64 nsamples; down(P.nsamples, sizeof(u64), &nsamples); std::vector<Seed> samples(nsamples); if (nsamples) down(P.samples, sizeof(Seed) * nsamples, samples.data());
    std::sort(samples.begin(), samples.end(), key);
    std::vector<int32_t> samplerows; for (auto& s : samples) for (int v : {s.f, s.h, s.d, s.a, s.g, s.ch}) samplerows.push_back(v);
    u64 minbits; down(P.minoo_bits, sizeof(u64), &minbits); double minOO = __builtin_bit_cast(double, minbits);
    writevec(out / "lag_frames.bin", lags); writevec(out / "numerators.bin", num); writevec(out / "numerators_blocks.bin", numb); writevec(out / "seed_denominators.bin", seedden);
    writevec(out / "origin_seed_denominators.bin", origin_den); writevec(out / "validation_primary_seeds.bin", samplerows);
    writevec(out / "fes_counts_halves.bin", hist); writevec(out / "fes_counts_blocks.bin", bhist); writevec(out / "validation_FES_counts.bin", testhist); writevec(out / "validation_frame_indices.bin", sampleframes);
    std::vector<int32_t> os32(os.begin(), os.end()), hs32(hs.begin(), hs.end()); writevec(out / "oxygen_coordinate_indices.bin", os32); writevec(out / "hydrogen_coordinate_indices.bin", hs32);
    {                                                 // structure outputs
      std::vector<unsigned int> dens(nde), ori(nor); std::vector<u64> dout((size_t)nblk * 25), oout((size_t)nblk * 5), pop((size_t)nblk * 25), hbd((size_t)nblk * 200), hba((size_t)nblk * 400), bfr(nblk);
      std::vector<short> tspec((size_t)nf * 4);
      down(P.dens, sizeof(unsigned int) * nde, dens.data()); down(P.ori, sizeof(unsigned int) * nor, ori.data()); down(P.dout, sizeof(u64) * dout.size(), dout.data());
      down(P.oout, sizeof(u64) * oout.size(), oout.data()); down(P.pop, sizeof(u64) * pop.size(), pop.data()); down(P.hbd, sizeof(u64) * hbd.size(), hbd.data());
      down(P.hba, sizeof(u64) * hba.size(), hba.data()); down(P.tspec, sizeof(short) * tspec.size(), tspec.data());
      for (int b = 0; b < nblk; ++b) {                // channel 2 = OH- mobile + Os-H (arithmetic sum)
        bfr[b] = (u64)std::min(BLK, nf - b * BLK); oout[(size_t)b * 5 + 2] = oout[(size_t)b * 5] + oout[(size_t)b * 5 + 1];
        for (size_t i = 0; i < (size_t)360 * 360; ++i) ori[((size_t)b * 5 + 2) * 129600 + i] = ori[((size_t)b * 5) * 129600 + i] + ori[((size_t)b * 5 + 1) * 129600 + i];
      }
      writevec(out / "density.u32", dens); writevec(out / "orientation.u32", ori); writevec(out / "density_outside.u64", dout); writevec(out / "orientation_outside.u64", oout);
      std::vector<double> mnum(2 * nmq); std::vector<u64> mcnt(nmq); down(dmnum, sizeof(double) * 2 * nmq, mnum.data()); down(dmcnt, sizeof(u64) * nmq, mcnt.data());
      writevec(out / "msd_lag_frames.i32", mlags); writevec(out / "msd_numerators.f64", mnum); writevec(out / "msd_counts.u64", mcnt);
      writevec(out / "population.u64", pop); writevec(out / "hbond_donated.u64", hbd); writevec(out / "hbond_accepted.u64", hba); writevec(out / "species.i16", tspec); writevec(out / "block_frames.u64", bfr);
    }
    double sec = now();
    std::ofstream meta(out / "kernel_summary.json"); meta << std::setprecision(17)
      << "{\n\"kernel\":\"waterint hb_correlation gpu\",\n\"frames\":" << nf << ",\n\"intervals\":" << nf - 1 << ",\n\"ring_frames\":" << NB << ",\n\"block_frames\":" << BLK << ",\n\"blocks\":" << nblk
      << ",\n\"unassigned_H_total\":" << stats[S_UNASSIGNED] << ",\n\"owner_ties\":" << stats[S_OWNER_TIES] << ",\n\"primary_ties\":" << stats[S_PRIMARY_TIES]
      << ",\n\"seed_records\":" << nseed_total << ",\n\"lag_count\":" << nl << ",\n\"origin_count\":" << norig << ",\n\"elapsed_seconds\":" << sec
      << ",\n\"elapsed_read_frame_corr_seconds\":[" << tread << "," << tframe << "," << tcorr << "]"
      << ",\n\"msd_stride_frames\":" << MS << ",\n\"msd_ring_frames\":" << MR << ",\n\"msd_lags\":" << nml << ",\n\"msd_atoms_mobileO_H\":[" << nnl << "," << nm - nnl << "],\n\"msd_seconds\":" << tmsd
      << ",\n\"FES_exposure\":[" << stats[S_FESX0] << "," << stats[S_FESX1] << "," << stats[S_FESX2] << "," << stats[S_FESX3]
      << "],\n\"FES_outside_grid\":[" << stats[S_FESO0] << "," << stats[S_FESO1] << "," << stats[S_FESO2] << "," << stats[S_FESO3]
      << "],\n\"eligible_edges\":[" << stats[S_ELIG0] << "," << stats[S_ELIG1] << "],\n\"primary_edges\":[" << stats[S_PRIM0] << "," << stats[S_PRIM1]
      << "],\n\"region_counts\":[" << stats[S_REG0] << "," << stats[S_REG0 + 1] << "," << stats[S_REG0 + 2] << "," << stats[S_REG0 + 3] << "," << stats[S_REG0 + 4] << "," << stats[S_REG0 + 5]
      << "],\n\"species_counts\":[" << stats[S_SPC0] << "," << stats[S_SPC0 + 1] << "," << stats[S_SPC0 + 2] << "," << stats[S_SPC0 + 3] << "," << stats[S_SPC0 + 4]
      << "],\n\"minimum_eligible_OO_A\":" << minOO << "\n}\n";
    std::cout << "COMPLETE frames=" << nf << " seconds=" << sec << " (read+upload " << tread << ", frames " << tframe << ", correlations " << tcorr << ")" << std::endl;
    return 0;
  } catch (const std::exception& e) { std::cerr << "ERROR " << e.what() << std::endl; return 2; }
}
