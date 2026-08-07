#include "../chemistry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr double RAD_TO_DEG = 57.2957795130823208768;

// Wrap one displacement component into the nearest periodic image.
double minimum_image_delta(double delta, double length, bool enabled) {
    return enabled ? delta - std::nearbyint(delta / length) * length : delta;
}

// Validate PBC input once and convert byte flags to the bool form used internally.
bool parse_pbc(
    const double* cell,
    const std::uint8_t* pbc,
    bool* use_pbc,
    double* cell_lengths
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        use_pbc[axis] = pbc != nullptr && pbc[axis] != 0;
        cell_lengths[axis] = 0.0;
        if (use_pbc[axis]) {
            if (cell == nullptr || cell[axis] <= 0.0) {
                return false;
            }
            cell_lengths[axis] = cell[axis];
        }
    }
    return true;
}

// Test the optional H-A distance and minimum D-H-A angle for one candidate.
bool passes_dha_geometry(
    const double* oxygen_positions,
    std::size_t donor_index,
    std::size_t acceptor_index,
    const double* hydrogen_positions,
    std::size_t hydrogen_index,
    double angle_min,
    double h_acceptor_cutoff,
    const double* cell_lengths,
    const bool* use_pbc,
    double* h_acceptor_distance
) {
    double donor_vector[3];
    double acceptor_vector[3];
    for (std::size_t axis = 0; axis < 3; ++axis) {
        donor_vector[axis] = minimum_image_delta(
            oxygen_positions[3 * donor_index + axis] - hydrogen_positions[3 * hydrogen_index + axis],
            cell_lengths[axis],
            use_pbc[axis]
        );
        acceptor_vector[axis] = minimum_image_delta(
            oxygen_positions[3 * acceptor_index + axis] - hydrogen_positions[3 * hydrogen_index + axis],
            cell_lengths[axis],
            use_pbc[axis]
        );
    }
    const double donor_norm2 =
        donor_vector[0] * donor_vector[0] + donor_vector[1] * donor_vector[1] + donor_vector[2] * donor_vector[2];
    const double acceptor_norm2 =
        acceptor_vector[0] * acceptor_vector[0] + acceptor_vector[1] * acceptor_vector[1] + acceptor_vector[2] * acceptor_vector[2];
    if (!(donor_norm2 > 0.0) || !(acceptor_norm2 > 0.0)) {
        return false;
    }
    const double distance = std::sqrt(acceptor_norm2);
    if (std::isfinite(h_acceptor_cutoff) && distance > h_acceptor_cutoff) {
        return false;
    }
    double cosine =
        (donor_vector[0] * acceptor_vector[0] + donor_vector[1] * acceptor_vector[1] + donor_vector[2] * acceptor_vector[2]) /
        std::sqrt(donor_norm2 * acceptor_norm2);
    cosine = std::max(-1.0, std::min(1.0, cosine));
    if (std::acos(cosine) * RAD_TO_DEG < angle_min) {
        return false;
    }
    *h_acceptor_distance = distance;
    return true;
}

}  // namespace

// Consume an O-H neighbor matrix and fill donated/accepted H-bond counts per O.
// Python later converts these integer counts into topology labels such as DDAA.
extern "C" int waterint_hbond_geometry_counts(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    const std::int64_t* hydrogen_counts,
    const std::int64_t* hydrogen_matrix,
    std::size_t hydrogen_capacity,
    double oo_cutoff,
    double dha_angle_min,
    double h_acceptor_cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    int max_acceptors_per_hydrogen,
    std::int64_t* donor_counts,
    std::int64_t* acceptor_counts
) {
    // The C ABI returns status codes because exceptions cannot safely cross ctypes.
    if (
        oxygen_positions == nullptr || hydrogen_positions == nullptr || hydrogen_counts == nullptr ||
        hydrogen_matrix == nullptr || donor_counts == nullptr || acceptor_counts == nullptr ||
        hydrogen_capacity == 0 || oo_cutoff <= 0.0 || dha_angle_min < 0.0 || dha_angle_min > 180.0
    ) {
        return 1;
    }

    bool use_pbc[3];
    double cell_lengths[3];
    if (!parse_pbc(cell, pbc, use_pbc, cell_lengths)) {
        return 2;
    }

    // One shared cell-list implementation supplies nearby oxygen acceptors.
    const waterint::CutoffNeighborSearch acceptor_search(
        oxygen_positions, n_oxygen, oxygen_positions, n_oxygen, oo_cutoff, cell_lengths, use_pbc
    );
    for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
        donor_counts[oxygen_index] = 0;
        acceptor_counts[oxygen_index] = 0;
    }

    // Outer loop: visit every possible donor oxygen.
    for (std::size_t donor_index = 0; donor_index < n_oxygen; ++donor_index) {
        const std::int64_t hydrogen_count = hydrogen_counts[donor_index];
        if (hydrogen_count < 0 || static_cast<std::size_t>(hydrogen_count) > hydrogen_capacity) {
            return 3;
        }
        if (hydrogen_count == 0) {
            continue;
        }
        // O-O cutoff is applied by the cell-list query before angle calculations.
        const std::vector<std::size_t> acceptors = acceptor_search.collect_indices(donor_index);
        if (acceptors.empty()) {
            continue;
        }
        const std::int64_t* hydrogen_row = hydrogen_matrix + donor_index * hydrogen_capacity;

        // Middle loop: test each H covalently attached to this donor oxygen.
        for (std::int64_t row_index = 0; row_index < hydrogen_count; ++row_index) {
            const std::int64_t raw_hydrogen_index = hydrogen_row[row_index];
            if (raw_hydrogen_index < 0 || static_cast<std::size_t>(raw_hydrogen_index) >= n_hydrogen) {
                return 4;
            }
            const std::size_t hydrogen_index = static_cast<std::size_t>(raw_hydrogen_index);
            bool has_best = false;
            double best_distance = std::numeric_limits<double>::infinity();
            std::size_t best_acceptor = 0;

            // Inner loop: apply H-A cutoff and D-H-A angle to nearby acceptors.
            for (std::size_t acceptor_index : acceptors) {
                if (acceptor_index == donor_index) {
                    continue;
                }
                double distance = 0.0;
                if (!passes_dha_geometry(
                        oxygen_positions,
                        donor_index,
                        acceptor_index,
                        hydrogen_positions,
                        hydrogen_index,
                        dha_angle_min,
                        h_acceptor_cutoff,
                        cell_lengths,
                        use_pbc,
                        &distance
                    )) {
                    continue;
                }
                if (max_acceptors_per_hydrogen != 0) {
                    // Candidate indices are sorted, preserving Python's first-index tie behavior.
                    if (!has_best || distance < best_distance) {
                        has_best = true;
                        best_distance = distance;
                        best_acceptor = acceptor_index;
                    }
                } else {
                    // Multi-acceptor mode counts every candidate that passes geometry.
                    donor_counts[donor_index] += 1;
                    acceptor_counts[acceptor_index] += 1;
                }
            }

            // Default mode records only the nearest passing acceptor for this H.
            if (max_acceptors_per_hydrogen != 0 && has_best) {
                donor_counts[donor_index] += 1;
                acceptor_counts[best_acceptor] += 1;
            }
        }
    }
    return 0;
}
