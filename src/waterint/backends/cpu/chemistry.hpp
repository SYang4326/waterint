#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace waterint {

// Reusable cell-list search for the geometry kernels in this package.
class CutoffNeighborSearch {
public:
    // Build candidate bins once so many query atoms can reuse the same spatial index.
    CutoffNeighborSearch(
        const double* query_positions,
        std::size_t n_queries,
        const double* candidate_positions,
        std::size_t n_candidates,
        double cutoff,
        const double* cell_lengths,
        const bool* use_pbc
    )
        : query_positions_(query_positions),
          n_queries_(n_queries),
          candidate_positions_(candidate_positions),
          n_candidates_(n_candidates),
          cutoff2_(cutoff * cutoff),
          cell_lengths_{cell_lengths[0], cell_lengths[1], cell_lengths[2]},
          use_pbc_{use_pbc[0], use_pbc[1], use_pbc[2]} {
        use_cell_list_ = build_search_box(cutoff, &box_);
        if (use_cell_list_) {
            const std::size_t total_bins = static_cast<std::size_t>(box_.bins[0]) *
                                           static_cast<std::size_t>(box_.bins[1]) *
                                           static_cast<std::size_t>(box_.bins[2]);
            bins_.resize(total_bins);
            for (std::size_t candidate_index = 0; candidate_index < n_candidates_; ++candidate_index) {
                const int ix = coordinate_bin(candidate_positions_[3 * candidate_index], 0, box_);
                const int iy = coordinate_bin(candidate_positions_[3 * candidate_index + 1], 1, box_);
                const int iz = coordinate_bin(candidate_positions_[3 * candidate_index + 2], 2, box_);
                bins_[flat_bin_index(ix, iy, iz, box_)].push_back(candidate_index);
            }
        }
    }

    // Count neighbors and optionally copy their candidate indices into a caller-owned row.
    std::int64_t collect(
        std::size_t query_index,
        std::int64_t* neighbor_row = nullptr,
        std::size_t neighbor_capacity = 0
    ) const {
        return use_cell_list_
            ? collect_cell_list(query_index, neighbor_row, neighbor_capacity)
            : collect_bruteforce(query_index, neighbor_row, neighbor_capacity);
    }

    // Return sorted candidate indices for algorithms that need neighbor identities.
    std::vector<std::size_t> collect_indices(std::size_t query_index) const {
        std::vector<std::size_t> indices;
        collect_indices(query_index, &indices);
        std::sort(indices.begin(), indices.end());
        return indices;
    }

private:
    struct SearchBox {
        double origin[3] = {0.0, 0.0, 0.0};
        double length[3] = {0.0, 0.0, 0.0};
        int bins[3] = {1, 1, 1};
        double bin_width[3] = {1.0, 1.0, 1.0};
        bool periodic[3] = {false, false, false};
    };

    // Apply the minimum-image convention to one Cartesian component.
    static double minimum_image_delta(double delta, double length, bool enabled) {
        return enabled ? delta - std::nearbyint(delta / length) * length : delta;
    }

    static int clamp_int(int value, int lower, int upper) {
        return std::max(lower, std::min(value, upper));
    }

    static int wrap_int(int value, int size) {
        const int wrapped = value % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }

    static std::size_t flat_bin_index(int ix, int iy, int iz, const SearchBox& box) {
        return static_cast<std::size_t>((iz * box.bins[1] + iy) * box.bins[0] + ix);
    }

    static int coordinate_bin(double value, int axis, const SearchBox& box) {
        double shifted = value - box.origin[axis];
        if (box.periodic[axis]) {
            shifted -= std::floor(shifted / box.length[axis]) * box.length[axis];
        }
        return clamp_int(
            static_cast<int>(std::floor(shifted / box.bin_width[axis])),
            0,
            box.bins[axis] - 1
        );
    }

    // Define periodic or data-derived bounds and choose bins at least cutoff wide.
    bool build_search_box(double cutoff, SearchBox* box) const {
        if (box == nullptr || cutoff <= 0.0) {
            return false;
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            box->periodic[axis] = use_pbc_[axis];
            if (use_pbc_[axis]) {
                box->origin[axis] = 0.0;
                box->length[axis] = cell_lengths_[axis];
            } else {
                bool initialized = false;
                double lo = 0.0;
                double hi = 0.0;
                for (std::size_t index = 0; index < n_queries_; ++index) {
                    const double value = query_positions_[3 * index + axis];
                    lo = initialized ? std::min(lo, value) : value;
                    hi = initialized ? std::max(hi, value) : value;
                    initialized = true;
                }
                for (std::size_t index = 0; index < n_candidates_; ++index) {
                    const double value = candidate_positions_[3 * index + axis];
                    lo = initialized ? std::min(lo, value) : value;
                    hi = initialized ? std::max(hi, value) : value;
                    initialized = true;
                }
                if (!initialized) {
                    return false;
                }
                box->origin[axis] = lo - cutoff;
                box->length[axis] = std::max(cutoff, (hi - lo) + 2.0 * cutoff);
            }
            box->bins[axis] = std::max(1, static_cast<int>(std::floor(box->length[axis] / cutoff)));
            box->bin_width[axis] = box->length[axis] / static_cast<double>(box->bins[axis]);
            if (box->bin_width[axis] <= 0.0) {
                return false;
            }
        }
        return true;
    }

    // Perform the exact squared-distance check after coarse bin filtering.
    bool is_neighbor(std::size_t query_index, std::size_t candidate_index) const {
        double dx = candidate_positions_[3 * candidate_index] - query_positions_[3 * query_index];
        double dy = candidate_positions_[3 * candidate_index + 1] - query_positions_[3 * query_index + 1];
        double dz = candidate_positions_[3 * candidate_index + 2] - query_positions_[3 * query_index + 2];
        dx = minimum_image_delta(dx, cell_lengths_[0], use_pbc_[0]);
        dy = minimum_image_delta(dy, cell_lengths_[1], use_pbc_[1]);
        dz = minimum_image_delta(dz, cell_lengths_[2], use_pbc_[2]);
        return dx * dx + dy * dy + dz * dz <= cutoff2_;
    }

    static void add_neighbor(
        std::int64_t* neighbor_row,
        std::size_t neighbor_capacity,
        std::int64_t* count,
        std::size_t candidate_index
    ) {
        if (neighbor_row != nullptr && static_cast<std::size_t>(*count) < neighbor_capacity) {
            neighbor_row[*count] = static_cast<std::int64_t>(candidate_index);
        }
        *count += 1;
    }

    // Fallback used only when a valid search box cannot be constructed.
    std::int64_t collect_bruteforce(
        std::size_t query_index,
        std::int64_t* neighbor_row,
        std::size_t neighbor_capacity
    ) const {
        std::int64_t count = 0;
        for (std::size_t candidate_index = 0; candidate_index < n_candidates_; ++candidate_index) {
            if (is_neighbor(query_index, candidate_index)) {
                add_neighbor(neighbor_row, neighbor_capacity, &count, candidate_index);
            }
        }
        return count;
    }

    // Search only the query atom's bin and its neighboring bins.
    std::int64_t collect_cell_list(
        std::size_t query_index,
        std::int64_t* neighbor_row,
        std::size_t neighbor_capacity
    ) const {
        const int center_x = coordinate_bin(query_positions_[3 * query_index], 0, box_);
        const int center_y = coordinate_bin(query_positions_[3 * query_index + 1], 1, box_);
        const int center_z = coordinate_bin(query_positions_[3 * query_index + 2], 2, box_);
        int x_bins[3] = {0, 0, 0};
        int y_bins[3] = {0, 0, 0};
        int z_bins[3] = {0, 0, 0};
        const int n_x = nearby_bins(center_x, 0, x_bins);
        const int n_y = nearby_bins(center_y, 1, y_bins);
        const int n_z = nearby_bins(center_z, 2, z_bins);
        std::int64_t count = 0;
        for (int z_i = 0; z_i < n_z; ++z_i) {
            for (int y_i = 0; y_i < n_y; ++y_i) {
                for (int x_i = 0; x_i < n_x; ++x_i) {
                    const std::vector<std::size_t>& candidates =
                        bins_[flat_bin_index(x_bins[x_i], y_bins[y_i], z_bins[z_i], box_)];
                    for (std::size_t candidate_index : candidates) {
                        if (is_neighbor(query_index, candidate_index)) {
                            add_neighbor(neighbor_row, neighbor_capacity, &count, candidate_index);
                        }
                    }
                }
            }
        }
        return count;
    }

    void collect_indices(std::size_t query_index, std::vector<std::size_t>* indices) const {
        if (indices == nullptr) {
            return;
        }
        if (!use_cell_list_) {
            for (std::size_t candidate_index = 0; candidate_index < n_candidates_; ++candidate_index) {
                if (is_neighbor(query_index, candidate_index)) {
                    indices->push_back(candidate_index);
                }
            }
            return;
        }
        const int center_x = coordinate_bin(query_positions_[3 * query_index], 0, box_);
        const int center_y = coordinate_bin(query_positions_[3 * query_index + 1], 1, box_);
        const int center_z = coordinate_bin(query_positions_[3 * query_index + 2], 2, box_);
        int x_bins[3] = {0, 0, 0};
        int y_bins[3] = {0, 0, 0};
        int z_bins[3] = {0, 0, 0};
        const int n_x = nearby_bins(center_x, 0, x_bins);
        const int n_y = nearby_bins(center_y, 1, y_bins);
        const int n_z = nearby_bins(center_z, 2, z_bins);
        for (int z_i = 0; z_i < n_z; ++z_i) {
            for (int y_i = 0; y_i < n_y; ++y_i) {
                for (int x_i = 0; x_i < n_x; ++x_i) {
                    const std::vector<std::size_t>& candidates =
                        bins_[flat_bin_index(x_bins[x_i], y_bins[y_i], z_bins[z_i], box_)];
                    for (std::size_t candidate_index : candidates) {
                        if (is_neighbor(query_index, candidate_index)) {
                            indices->push_back(candidate_index);
                        }
                    }
                }
            }
        }
    }

    // Return unique neighboring bin indices, wrapping periodic boundaries.
    int nearby_bins(int center, int axis, int* values) const {
        int count = 0;
        for (int delta = -1; delta <= 1; ++delta) {
            int bin = center + delta;
            if (box_.periodic[axis]) {
                bin = wrap_int(bin, box_.bins[axis]);
            } else if (bin < 0 || bin >= box_.bins[axis]) {
                continue;
            }
            bool seen = false;
            for (int index = 0; index < count; ++index) {
                seen = seen || values[index] == bin;
            }
            if (!seen) {
                values[count] = bin;
                count += 1;
            }
        }
        return count;
    }

    const double* query_positions_;
    std::size_t n_queries_;
    const double* candidate_positions_;
    std::size_t n_candidates_;
    double cutoff2_;
    double cell_lengths_[3];
    bool use_pbc_[3];
    bool use_cell_list_ = false;
    SearchBox box_;
    std::vector<std::vector<std::size_t>> bins_;
};

}  // namespace waterint
