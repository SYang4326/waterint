#include "chemistry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Normalize optional PBC pointers for the shared C++ neighbor-search API.
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

// Check the arrays, cutoff, and periodic cell before constructing a cell list.
int validate_common_inputs(
    const double* oxygen_positions,
    const double* hydrogen_positions,
    double cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    bool* use_pbc,
    double* cell_lengths
) {
    if (oxygen_positions == nullptr || hydrogen_positions == nullptr || cutoff <= 0.0) {
        return 1;
    }
    return parse_pbc(cell, pbc, use_pbc, cell_lengths) ? 0 : 2;
}

bool invert_3x3(const double* matrix, double* inverse) {
    const double a = matrix[0];
    const double b = matrix[1];
    const double c = matrix[2];
    const double d = matrix[3];
    const double e = matrix[4];
    const double f = matrix[5];
    const double g = matrix[6];
    const double h = matrix[7];
    const double i = matrix[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::abs(det) <= 1.0e-14) {
        return false;
    }
    const double inv_det = 1.0 / det;
    inverse[0] = (e * i - f * h) * inv_det;
    inverse[1] = (c * h - b * i) * inv_det;
    inverse[2] = (b * f - c * e) * inv_det;
    inverse[3] = (f * g - d * i) * inv_det;
    inverse[4] = (a * i - c * g) * inv_det;
    inverse[5] = (c * d - a * f) * inv_det;
    inverse[6] = (d * h - e * g) * inv_det;
    inverse[7] = (b * g - a * h) * inv_det;
    inverse[8] = (a * e - b * d) * inv_det;
    return true;
}

double distance2_full_cell(
    const double* from,
    const double* to,
    const double* cell_vectors,
    const double* inverse_cell_vectors,
    const std::uint8_t* pbc
) {
    double dx = to[0] - from[0];
    double dy = to[1] - from[1];
    double dz = to[2] - from[2];
    double sx = dx * inverse_cell_vectors[0] + dy * inverse_cell_vectors[3] + dz * inverse_cell_vectors[6];
    double sy = dx * inverse_cell_vectors[1] + dy * inverse_cell_vectors[4] + dz * inverse_cell_vectors[7];
    double sz = dx * inverse_cell_vectors[2] + dy * inverse_cell_vectors[5] + dz * inverse_cell_vectors[8];
    if (pbc != nullptr && pbc[0] != 0) sx -= std::nearbyint(sx);
    if (pbc != nullptr && pbc[1] != 0) sy -= std::nearbyint(sy);
    if (pbc != nullptr && pbc[2] != 0) sz -= std::nearbyint(sz);
    dx = sx * cell_vectors[0] + sy * cell_vectors[3] + sz * cell_vectors[6];
    dy = sx * cell_vectors[1] + sy * cell_vectors[4] + sz * cell_vectors[7];
    dz = sx * cell_vectors[2] + sy * cell_vectors[5] + sz * cell_vectors[8];
    return dx * dx + dy * dy + dz * dz;
}

void group_h_counts(
    const std::int64_t* oxygen_indices,
    const std::int64_t* h_counts,
    std::size_t n_oxygen,
    std::int64_t* label_counts,
    std::int64_t* grouped_indices
) {
    for (std::size_t label = 0; label < 5; ++label) {
        label_counts[label] = 0;
    }
    for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
        const std::int64_t h_count = h_counts[oxygen_index];
        const std::size_t label = h_count >= 0 && h_count <= 3 ? static_cast<std::size_t>(h_count) : 4;
        grouped_indices[label * n_oxygen + static_cast<std::size_t>(label_counts[label])] = oxygen_indices[oxygen_index];
        label_counts[label] += 1;
    }
}

}  // namespace

// Count attached H atoms only; density species classification uses this compact path.
extern "C" int waterint_count_hydrogen_neighbors(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    double cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    std::int64_t* counts
) {
    bool use_pbc[3];
    double cell_lengths[3];
    const int status = validate_common_inputs(
        oxygen_positions, hydrogen_positions, cutoff, cell, pbc, use_pbc, cell_lengths
    );
    if (status != 0 || counts == nullptr) {
        return status == 0 ? 1 : status;
    }
    const waterint::CutoffNeighborSearch search(
        oxygen_positions, n_oxygen, hydrogen_positions, n_hydrogen, cutoff, cell_lengths, use_pbc
    );
    for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
        counts[oxygen_index] = search.collect(oxygen_index);
    }
    return 0;
}

extern "C" int waterint_classify_oxygen_by_h_count_nearest(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    const std::int64_t* oxygen_indices,
    double cutoff,
    const double* cell,
    const double* cell_vectors,
    const std::uint8_t* pbc,
    std::int64_t* label_counts,
    std::int64_t* grouped_indices
) {
    if (
        oxygen_positions == nullptr || hydrogen_positions == nullptr || oxygen_indices == nullptr ||
        label_counts == nullptr || grouped_indices == nullptr || cutoff <= 0.0
    ) {
        return 1;
    }

    std::vector<std::int64_t> h_counts(n_oxygen, 0);
    const double cutoff2 = cutoff * cutoff;
    if (cell_vectors != nullptr) {
        double inverse_cell_vectors[9];
        if (!invert_3x3(cell_vectors, inverse_cell_vectors)) {
            return 2;
        }
        for (std::size_t hydrogen_index = 0; hydrogen_index < n_hydrogen; ++hydrogen_index) {
            const double* hydrogen = hydrogen_positions + 3 * hydrogen_index;
            double best_distance2 = cutoff2;
            std::size_t best_oxygen = n_oxygen;
            for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
                const double distance2 = distance2_full_cell(
                    oxygen_positions + 3 * oxygen_index,
                    hydrogen,
                    cell_vectors,
                    inverse_cell_vectors,
                    pbc
                );
                if (distance2 <= best_distance2) {
                    best_distance2 = distance2;
                    best_oxygen = oxygen_index;
                }
            }
            if (best_oxygen < n_oxygen) {
                h_counts[best_oxygen] += 1;
            }
        }
        group_h_counts(oxygen_indices, h_counts.data(), n_oxygen, label_counts, grouped_indices);
        return 0;
    }

    bool use_pbc[3];
    double cell_lengths[3];
    const int status = validate_common_inputs(
        oxygen_positions, hydrogen_positions, cutoff, cell, pbc, use_pbc, cell_lengths
    );
    if (status != 0) {
        return status;
    }
    const waterint::CutoffNeighborSearch search(
        hydrogen_positions, n_hydrogen, oxygen_positions, n_oxygen, cutoff, cell_lengths, use_pbc
    );
    for (std::size_t hydrogen_index = 0; hydrogen_index < n_hydrogen; ++hydrogen_index) {
        std::vector<std::size_t> candidates = search.collect_indices(hydrogen_index);
        double best_distance2 = cutoff2;
        std::size_t best_oxygen = n_oxygen;
        for (std::size_t oxygen_index : candidates) {
            double dx = oxygen_positions[3 * oxygen_index] - hydrogen_positions[3 * hydrogen_index];
            double dy = oxygen_positions[3 * oxygen_index + 1] - hydrogen_positions[3 * hydrogen_index + 1];
            double dz = oxygen_positions[3 * oxygen_index + 2] - hydrogen_positions[3 * hydrogen_index + 2];
            dx = use_pbc[0] ? dx - std::nearbyint(dx / cell_lengths[0]) * cell_lengths[0] : dx;
            dy = use_pbc[1] ? dy - std::nearbyint(dy / cell_lengths[1]) * cell_lengths[1] : dy;
            dz = use_pbc[2] ? dz - std::nearbyint(dz / cell_lengths[2]) * cell_lengths[2] : dz;
            const double distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= best_distance2) {
                best_distance2 = distance2;
                best_oxygen = oxygen_index;
            }
        }
        if (best_oxygen < n_oxygen) {
            h_counts[best_oxygen] += 1;
        }
    }
    group_h_counts(oxygen_indices, h_counts.data(), n_oxygen, label_counts, grouped_indices);
    return 0;
}

// Group global oxygen indices by attached-H count without creating Python records.
extern "C" int waterint_classify_oxygen_by_h_count_compact(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    const std::int64_t* oxygen_indices,
    double cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    std::int64_t* label_counts,
    std::int64_t* grouped_indices
) {
    bool use_pbc[3];
    double cell_lengths[3];
    const int status = validate_common_inputs(
        oxygen_positions, hydrogen_positions, cutoff, cell, pbc, use_pbc, cell_lengths
    );
    if (status != 0 || oxygen_indices == nullptr || label_counts == nullptr || grouped_indices == nullptr) {
        return status == 0 ? 1 : status;
    }
    for (std::size_t label = 0; label < 5; ++label) {
        label_counts[label] = 0;
    }
    const waterint::CutoffNeighborSearch search(
        oxygen_positions, n_oxygen, hydrogen_positions, n_hydrogen, cutoff, cell_lengths, use_pbc
    );
    for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
        const std::int64_t h_count = search.collect(oxygen_index);
        const std::size_t label = h_count >= 0 && h_count <= 3 ? static_cast<std::size_t>(h_count) : 4;
        grouped_indices[label * n_oxygen + static_cast<std::size_t>(label_counts[label])] = oxygen_indices[oxygen_index];
        label_counts[label] += 1;
    }
    return 0;
}

// Return both attached-H counts and local H identities for orientation and H-bond kernels.
extern "C" int waterint_hydrogen_neighbor_matrix(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    double cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    std::size_t neighbor_capacity,
    std::int64_t* neighbor_counts,
    std::int64_t* neighbor_matrix,
    std::size_t* required_capacity
) {
    bool use_pbc[3];
    double cell_lengths[3];
    const int status = validate_common_inputs(
        oxygen_positions, hydrogen_positions, cutoff, cell, pbc, use_pbc, cell_lengths
    );
    if (
        status != 0 || neighbor_counts == nullptr || neighbor_matrix == nullptr ||
        required_capacity == nullptr || neighbor_capacity == 0
    ) {
        return status == 0 ? 1 : status;
    }
    const waterint::CutoffNeighborSearch search(
        oxygen_positions, n_oxygen, hydrogen_positions, n_hydrogen, cutoff, cell_lengths, use_pbc
    );
    std::size_t max_neighbors = 0;
    for (std::size_t oxygen_index = 0; oxygen_index < n_oxygen; ++oxygen_index) {
        std::int64_t* row = neighbor_matrix + oxygen_index * neighbor_capacity;
        const std::int64_t count = search.collect(oxygen_index, row, neighbor_capacity);
        neighbor_counts[oxygen_index] = count;
        if (count > 0 && static_cast<std::size_t>(count) <= neighbor_capacity) {
            std::sort(row, row + count);
        }
        max_neighbors = std::max(max_neighbors, static_cast<std::size_t>(count));
    }
    *required_capacity = max_neighbors;
    return max_neighbors > neighbor_capacity ? 3 : 0;
}
