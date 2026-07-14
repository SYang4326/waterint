#include "chemistry.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
