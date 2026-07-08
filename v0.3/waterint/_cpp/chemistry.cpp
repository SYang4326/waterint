#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct SearchBox {
    double origin[3] = {0.0, 0.0, 0.0};
    double length[3] = {0.0, 0.0, 0.0};
    int bins[3] = {1, 1, 1};
    double bin_width[3] = {1.0, 1.0, 1.0};
    bool periodic[3] = {false, false, false};
};

double minimum_image_delta(double delta, double length, bool enabled) {
    if (!enabled) {
        return delta;
    }
    return delta - std::nearbyint(delta / length) * length;
}

int clamp_int(int value, int lower, int upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

int wrap_int(int value, int size) {
    int wrapped = value % size;
    if (wrapped < 0) {
        wrapped += size;
    }
    return wrapped;
}

std::size_t flat_bin_index(int ix, int iy, int iz, const SearchBox& box) {
    return static_cast<std::size_t>((iz * box.bins[1] + iy) * box.bins[0] + ix);
}

int coordinate_bin(double value, int axis, const SearchBox& box) {
    double shifted = value - box.origin[axis];
    if (box.periodic[axis]) {
        shifted = shifted - std::floor(shifted / box.length[axis]) * box.length[axis];
    }
    const int raw = static_cast<int>(std::floor(shifted / box.bin_width[axis]));
    return clamp_int(raw, 0, box.bins[axis] - 1);
}

bool build_search_box(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    double cutoff,
    const double* cell,
    const bool* use_pbc,
    SearchBox* box
) {
    if (box == nullptr || cutoff <= 0.0) {
        return false;
    }

    for (std::size_t axis = 0; axis < 3; ++axis) {
        box->periodic[axis] = use_pbc[axis];
        if (use_pbc[axis]) {
            if (cell == nullptr || cell[axis] <= 0.0) {
                return false;
            }
            box->origin[axis] = 0.0;
            box->length[axis] = cell[axis];
        } else {
            double lo = 0.0;
            double hi = 0.0;
            bool initialized = false;
            for (std::size_t i = 0; i < n_oxygen; ++i) {
                const double value = oxygen_positions[3 * i + axis];
                if (!initialized) {
                    lo = value;
                    hi = value;
                    initialized = true;
                } else {
                    if (value < lo) {
                        lo = value;
                    }
                    if (value > hi) {
                        hi = value;
                    }
                }
            }
            for (std::size_t i = 0; i < n_hydrogen; ++i) {
                const double value = hydrogen_positions[3 * i + axis];
                if (!initialized) {
                    lo = value;
                    hi = value;
                    initialized = true;
                } else {
                    if (value < lo) {
                        lo = value;
                    }
                    if (value > hi) {
                        hi = value;
                    }
                }
            }
            box->origin[axis] = lo - cutoff;
            box->length[axis] = (hi - lo) + 2.0 * cutoff;
            if (box->length[axis] <= 0.0) {
                box->length[axis] = cutoff;
            }
        }
        box->bins[axis] = static_cast<int>(std::floor(box->length[axis] / cutoff));
        if (box->bins[axis] < 1) {
            box->bins[axis] = 1;
        }
        box->bin_width[axis] = box->length[axis] / static_cast<double>(box->bins[axis]);
        if (box->bin_width[axis] <= 0.0) {
            return false;
        }
    }
    return true;
}

std::int64_t count_hydrogen_neighbors_cell_list(
    const double* oxygen_positions,
    std::size_t oxygen_index,
    const double* hydrogen_positions,
    const std::vector<std::vector<std::size_t>>& bins,
    const SearchBox& box,
    double cutoff2
) {
    const double ox = oxygen_positions[3 * oxygen_index + 0];
    const double oy = oxygen_positions[3 * oxygen_index + 1];
    const double oz = oxygen_positions[3 * oxygen_index + 2];
    const int center_x = coordinate_bin(ox, 0, box);
    const int center_y = coordinate_bin(oy, 1, box);
    const int center_z = coordinate_bin(oz, 2, box);
    int x_bins[3] = {0, 0, 0};
    int y_bins[3] = {0, 0, 0};
    int z_bins[3] = {0, 0, 0};
    int n_x_bins = 0;
    int n_y_bins = 0;
    int n_z_bins = 0;
    auto add_unique = [](int* values, int* count, int value) {
        for (int i = 0; i < *count; ++i) {
            if (values[i] == value) {
                return;
            }
        }
        values[*count] = value;
        *count += 1;
    };
    for (int delta = -1; delta <= 1; ++delta) {
        int ix = center_x + delta;
        if (box.periodic[0]) {
            ix = wrap_int(ix, box.bins[0]);
        } else if (ix < 0 || ix >= box.bins[0]) {
            continue;
        }
        add_unique(x_bins, &n_x_bins, ix);
    }
    for (int delta = -1; delta <= 1; ++delta) {
        int iy = center_y + delta;
        if (box.periodic[1]) {
            iy = wrap_int(iy, box.bins[1]);
        } else if (iy < 0 || iy >= box.bins[1]) {
            continue;
        }
        add_unique(y_bins, &n_y_bins, iy);
    }
    for (int delta = -1; delta <= 1; ++delta) {
        int iz = center_z + delta;
        if (box.periodic[2]) {
            iz = wrap_int(iz, box.bins[2]);
        } else if (iz < 0 || iz >= box.bins[2]) {
            continue;
        }
        add_unique(z_bins, &n_z_bins, iz);
    }
    std::int64_t count = 0;

    for (int z_i = 0; z_i < n_z_bins; ++z_i) {
        const int iz = z_bins[z_i];
        for (int y_i = 0; y_i < n_y_bins; ++y_i) {
            const int iy = y_bins[y_i];
            for (int x_i = 0; x_i < n_x_bins; ++x_i) {
                const int ix = x_bins[x_i];
                const std::vector<std::size_t>& candidates = bins[flat_bin_index(ix, iy, iz, box)];
                for (std::size_t hydrogen_index : candidates) {
                    double dx = hydrogen_positions[3 * hydrogen_index + 0] - ox;
                    double dy = hydrogen_positions[3 * hydrogen_index + 1] - oy;
                    double dz = hydrogen_positions[3 * hydrogen_index + 2] - oz;
                    dx = minimum_image_delta(dx, box.length[0], box.periodic[0]);
                    dy = minimum_image_delta(dy, box.length[1], box.periodic[1]);
                    dz = minimum_image_delta(dz, box.length[2], box.periodic[2]);
                    const double distance2 = dx * dx + dy * dy + dz * dz;
                    if (distance2 <= cutoff2) {
                        ++count;
                    }
                }
            }
        }
    }
    return count;
}

}  // namespace

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
    if (oxygen_positions == nullptr || hydrogen_positions == nullptr || counts == nullptr || cutoff <= 0.0) {
        return 1;
    }

    bool use_pbc[3] = {false, false, false};
    double cell_lengths[3] = {0.0, 0.0, 0.0};
    if (pbc != nullptr) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            use_pbc[axis] = pbc[axis] != 0;
            if (use_pbc[axis]) {
                if (cell == nullptr || cell[axis] <= 0.0) {
                    return 2;
                }
                cell_lengths[axis] = cell[axis];
            }
        }
    }

    const double cutoff2 = cutoff * cutoff;

    SearchBox search_box;
    if (build_search_box(oxygen_positions, n_oxygen, hydrogen_positions, n_hydrogen, cutoff, cell_lengths, use_pbc, &search_box)) {
        const std::size_t total_bins = static_cast<std::size_t>(search_box.bins[0]) *
                                       static_cast<std::size_t>(search_box.bins[1]) *
                                       static_cast<std::size_t>(search_box.bins[2]);
        std::vector<std::vector<std::size_t>> bins(total_bins);
        for (std::size_t j = 0; j < n_hydrogen; ++j) {
            const int ix = coordinate_bin(hydrogen_positions[3 * j + 0], 0, search_box);
            const int iy = coordinate_bin(hydrogen_positions[3 * j + 1], 1, search_box);
            const int iz = coordinate_bin(hydrogen_positions[3 * j + 2], 2, search_box);
            bins[flat_bin_index(ix, iy, iz, search_box)].push_back(j);
        }

        for (std::size_t i = 0; i < n_oxygen; ++i) {
            counts[i] = count_hydrogen_neighbors_cell_list(oxygen_positions, i, hydrogen_positions, bins, search_box, cutoff2);
        }
        return 0;
    }

    for (std::size_t i = 0; i < n_oxygen; ++i) {
        std::int64_t count = 0;
        const double ox = oxygen_positions[3 * i + 0];
        const double oy = oxygen_positions[3 * i + 1];
        const double oz = oxygen_positions[3 * i + 2];
        for (std::size_t j = 0; j < n_hydrogen; ++j) {
            double dx = hydrogen_positions[3 * j + 0] - ox;
            double dy = hydrogen_positions[3 * j + 1] - oy;
            double dz = hydrogen_positions[3 * j + 2] - oz;
            dx = minimum_image_delta(dx, cell_lengths[0], use_pbc[0]);
            dy = minimum_image_delta(dy, cell_lengths[1], use_pbc[1]);
            dz = minimum_image_delta(dz, cell_lengths[2], use_pbc[2]);
            const double distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= cutoff2) {
                ++count;
            }
        }
        counts[i] = count;
    }

    return 0;
}

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
    if (
        oxygen_positions == nullptr ||
        hydrogen_positions == nullptr ||
        oxygen_indices == nullptr ||
        label_counts == nullptr ||
        grouped_indices == nullptr ||
        cutoff <= 0.0
    ) {
        return 1;
    }

    bool use_pbc[3] = {false, false, false};
    double cell_lengths[3] = {0.0, 0.0, 0.0};
    if (pbc != nullptr) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            use_pbc[axis] = pbc[axis] != 0;
            if (use_pbc[axis]) {
                if (cell == nullptr || cell[axis] <= 0.0) {
                    return 2;
                }
                cell_lengths[axis] = cell[axis];
            }
        }
    }

    for (std::size_t label = 0; label < 5; ++label) {
        label_counts[label] = 0;
    }

    const double cutoff2 = cutoff * cutoff;
    SearchBox search_box;
    if (build_search_box(oxygen_positions, n_oxygen, hydrogen_positions, n_hydrogen, cutoff, cell_lengths, use_pbc, &search_box)) {
        const std::size_t total_bins = static_cast<std::size_t>(search_box.bins[0]) *
                                       static_cast<std::size_t>(search_box.bins[1]) *
                                       static_cast<std::size_t>(search_box.bins[2]);
        std::vector<std::vector<std::size_t>> bins(total_bins);
        for (std::size_t j = 0; j < n_hydrogen; ++j) {
            const int ix = coordinate_bin(hydrogen_positions[3 * j + 0], 0, search_box);
            const int iy = coordinate_bin(hydrogen_positions[3 * j + 1], 1, search_box);
            const int iz = coordinate_bin(hydrogen_positions[3 * j + 2], 2, search_box);
            bins[flat_bin_index(ix, iy, iz, search_box)].push_back(j);
        }

        for (std::size_t i = 0; i < n_oxygen; ++i) {
            const std::int64_t h_count = count_hydrogen_neighbors_cell_list(
                oxygen_positions,
                i,
                hydrogen_positions,
                bins,
                search_box,
                cutoff2
            );
            std::size_t label = 4;
            if (h_count >= 0 && h_count <= 3) {
                label = static_cast<std::size_t>(h_count);
            }
            const std::int64_t offset = label_counts[label];
            grouped_indices[label * n_oxygen + static_cast<std::size_t>(offset)] = oxygen_indices[i];
            label_counts[label] += 1;
        }
        return 0;
    }

    for (std::size_t i = 0; i < n_oxygen; ++i) {
        std::int64_t h_count = 0;
        const double ox = oxygen_positions[3 * i + 0];
        const double oy = oxygen_positions[3 * i + 1];
        const double oz = oxygen_positions[3 * i + 2];
        for (std::size_t j = 0; j < n_hydrogen; ++j) {
            double dx = hydrogen_positions[3 * j + 0] - ox;
            double dy = hydrogen_positions[3 * j + 1] - oy;
            double dz = hydrogen_positions[3 * j + 2] - oz;
            dx = minimum_image_delta(dx, cell_lengths[0], use_pbc[0]);
            dy = minimum_image_delta(dy, cell_lengths[1], use_pbc[1]);
            dz = minimum_image_delta(dz, cell_lengths[2], use_pbc[2]);
            const double distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= cutoff2) {
                ++h_count;
            }
        }

        std::size_t label = 4;
        if (h_count >= 0 && h_count <= 3) {
            label = static_cast<std::size_t>(h_count);
        }
        const std::int64_t offset = label_counts[label];
        grouped_indices[label * n_oxygen + static_cast<std::size_t>(offset)] = oxygen_indices[i];
        label_counts[label] += 1;
    }

    return 0;
}
