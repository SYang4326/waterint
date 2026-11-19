// Exported entry of the canonical ownership rule (see ownership.hpp); used by waterint.prep.ownership.
#include "ownership.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

bool inv3(const double* m, double* inv) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::abs(det) <= 1.0e-14) return false;
    const double r = 1.0 / det;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * r; inv[1] = (m[2] * m[7] - m[1] * m[8]) * r; inv[2] = (m[1] * m[5] - m[2] * m[4]) * r;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * r; inv[4] = (m[0] * m[8] - m[2] * m[6]) * r; inv[5] = (m[2] * m[3] - m[0] * m[5]) * r;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * r; inv[7] = (m[1] * m[6] - m[0] * m[7]) * r; inv[8] = (m[0] * m[4] - m[1] * m[3]) * r;
    return true;
}

double dist_full(const double* a, const double* b, const double* h, const double* hi, const bool* p) {
    double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    double s[3];
    for (int k = 0; k < 3; ++k) s[k] = d[0] * hi[k] + d[1] * hi[3 + k] + d[2] * hi[6 + k];
    for (int k = 0; k < 3; ++k) if (p[k]) s[k] -= std::nearbyint(s[k]);
    double out = 0.0;
    for (int k = 0; k < 3; ++k) { const double x = s[0] * h[k] + s[1] * h[3 + k] + s[2] * h[6 + k]; out += x * x; }
    return std::sqrt(out);
}

}  // namespace

extern "C" int waterint_nearest_owner(
    const double* acceptors, std::size_t n_acceptors,
    const double* protons, std::size_t n_protons,
    double cutoff,                   // A; +inf = no cutoff
    double tie_tolerance,            // A
    int fallback_nearest,            // 1: protons without a candidate take the nearest acceptor (no cutoff)
    const double* cell_lengths,      // orthorhombic lengths (may be null if no pbc or cell_vectors given)
    const double* cell_vectors,      // 3x3 row-vector cell matrix, or null
    const std::uint8_t* pbc,         // 3 flags, or null for no periodic images
    const std::int64_t* ids,         // acceptor atom IDs for ties (null: index order)
    std::int64_t* owner,             // out [n_protons]: local acceptor index or -1
    std::int32_t* tied               // out [n_protons] (nullable): size of the tie set
) {
    if (acceptors == nullptr || protons == nullptr || owner == nullptr || !(cutoff > 0.0) || tie_tolerance < 0.0) return 1;
    bool use_pbc[3] = {pbc != nullptr && pbc[0] != 0, pbc != nullptr && pbc[1] != 0, pbc != nullptr && pbc[2] != 0};
    const bool any_pbc = use_pbc[0] || use_pbc[1] || use_pbc[2];
    if (cell_vectors != nullptr && any_pbc) {
        double hinv[9];
        if (!inv3(cell_vectors, hinv)) return 2;
        std::vector<std::size_t> cand;
        std::vector<double> dist;
        for (std::size_t h = 0; h < n_protons; ++h) {
            cand.clear(); dist.clear();
            std::vector<std::size_t> all;
            std::vector<double> alld;
            for (std::size_t a = 0; a < n_acceptors; ++a) {
                const double d = dist_full(acceptors + 3 * a, protons + 3 * h, cell_vectors, hinv, use_pbc);
                if (d <= cutoff) { cand.push_back(a); dist.push_back(d); }
                if (fallback_nearest) { all.push_back(a); alld.push_back(d); }
            }
            int t = 0;
            owner[h] = waterint::pick_owner(cand, dist, tie_tolerance, ids, &t);
            if (owner[h] < 0 && fallback_nearest && !all.empty()) owner[h] = waterint::pick_owner(all, alld, tie_tolerance, ids, &t);
            if (tied) tied[h] = t;
        }
        return 0;
    }
    double lengths[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        if (use_pbc[k]) {
            if (cell_lengths == nullptr || cell_lengths[k] <= 0.0) return 2;
            lengths[k] = cell_lengths[k];
        }
    }
    waterint::nearest_owners(acceptors, n_acceptors, protons, n_protons, cutoff, fallback_nearest != 0, tie_tolerance,
                             lengths, use_pbc, ids, owner, nullptr, tied);
    return 0;
}
