// The ONE proton -> acceptor ownership rule of WaterInt (prep.ownership, SFG, relabel):
//
//   candidates = acceptors with minimum-image distance d <= cutoff
//   d_min      = min d over the candidates          (no candidate -> free proton, owner -1)
//   tie set    = candidates with d <= d_min + tie_tolerance   (default 1e-8 A, profile h_assignment.tie_tolerance_A)
//   owner      = the tie-set acceptor with the lowest atom ID (acceptor index order when no IDs are given)
//
// `fallback_nearest` (SFG: every H must belong to one O-H bond): a proton without a candidate takes its nearest
// acceptor without a cutoff (same tie rule). The set definition makes the result independent of traversal order.
#pragma once

#include "chemistry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace waterint {

constexpr double kDefaultTieToleranceA = 1.0e-8;

inline double ortho_distance(const double* a, const double* b, const double* lengths, const bool* pbc) {
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        double x = a[k] - b[k];
        if (pbc[k]) x -= std::nearbyint(x / lengths[k]) * lengths[k];
        d2 += x * x;
    }
    return std::sqrt(d2);
}

// Pick the owner among candidate acceptors (their distances given); returns -1 if none.
inline std::int64_t pick_owner(const std::vector<std::size_t>& cand, const std::vector<double>& dist, double tol,
                               const std::int64_t* ids, int* n_tied) {
    double dmin = std::numeric_limits<double>::infinity();
    for (double d : dist) dmin = d < dmin ? d : dmin;
    std::int64_t best = -1;
    int ties = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
        if (dist[i] <= dmin + tol) {
            ++ties;
            const std::int64_t a = static_cast<std::int64_t>(cand[i]);
            if (best < 0 || (ids ? ids[a] < ids[best] : a < best)) best = a;
        }
    }
    if (n_tied) *n_tied = ties;
    return best;
}

// Orthorhombic / open-boundary version (cell list for a finite cutoff).
// owner[np]: local acceptor index or -1; n_within[np] (optional): acceptors within the cutoff;
// tied[np] (optional): size of the tie set (>1 = tie resolved by ID).
inline void nearest_owners(const double* acceptors, std::size_t na, const double* protons, std::size_t np,
                           double cutoff, bool fallback_nearest, double tol, const double* lengths, const bool* pbc,
                           const std::int64_t* ids, std::int64_t* owner, std::int64_t* n_within, std::int32_t* tied) {
    std::vector<std::size_t> cand;
    std::vector<double> dist;
    const bool finite = std::isfinite(cutoff);
    double safe_lengths[3] = {lengths ? lengths[0] : 0.0, lengths ? lengths[1] : 0.0, lengths ? lengths[2] : 0.0};
    if (finite) {
        const CutoffNeighborSearch search(protons, np, acceptors, na, cutoff, safe_lengths, pbc);
        for (std::size_t h = 0; h < np; ++h) {
            cand.clear(); dist.clear();
            for (std::size_t a : search.collect_indices(h)) {
                const double d = ortho_distance(acceptors + 3 * a, protons + 3 * h, safe_lengths, pbc);
                if (d <= cutoff) { cand.push_back(a); dist.push_back(d); }
            }
            int t = 0;
            owner[h] = pick_owner(cand, dist, tol, ids, &t);
            if (n_within) n_within[h] = static_cast<std::int64_t>(cand.size());
            if (tied) tied[h] = t;
        }
    } else {
        for (std::size_t h = 0; h < np; ++h) owner[h] = -1;
        if (n_within) for (std::size_t h = 0; h < np; ++h) n_within[h] = 0;
        if (tied) for (std::size_t h = 0; h < np; ++h) tied[h] = 0;
        fallback_nearest = true;
    }
    if (!fallback_nearest || na == 0) return;
    for (std::size_t h = 0; h < np; ++h) {
        if (owner[h] >= 0) continue;
        cand.clear(); dist.clear();
        for (std::size_t a = 0; a < na; ++a) {
            cand.push_back(a);
            dist.push_back(ortho_distance(acceptors + 3 * a, protons + 3 * h, safe_lengths, pbc));
        }
        int t = 0;
        owner[h] = pick_owner(cand, dist, tol, ids, &t);
        if (tied) tied[h] = t;
    }
}

}  // namespace waterint
