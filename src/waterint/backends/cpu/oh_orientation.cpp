#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr double RAD_TO_DEG = 57.2957795130823208768;

std::int64_t find_bin(double value, const double* edges, std::size_t n_bins) {
    if (!std::isfinite(value) || edges == nullptr || n_bins == 0) {
        return -1;
    }
    const double lower = edges[0];
    const double upper = edges[n_bins];
    if (value < lower || value > upper) {
        return -1;
    }
    if (value == upper) {
        return static_cast<std::int64_t>(n_bins - 1);
    }
    const double* upper_edge = std::upper_bound(edges, edges + n_bins + 1, value);
    if (upper_edge == edges || upper_edge == edges + n_bins + 1) {
        return -1;
    }
    return static_cast<std::int64_t>((upper_edge - edges) - 1);
}

int species_slot(std::int64_t hydrogen_count) {
    if (hydrogen_count < 1 || hydrogen_count > 3) {
        return -1;
    }
    return static_cast<int>(hydrogen_count - 1);
}

void add_histogram_sample(
    int slot,
    double z_value,
    double angle,
    const double* z_edges,
    std::size_t n_z_bins,
    const double* angle_edges,
    std::size_t n_angle_bins,
    double* histograms
) {
    const std::int64_t z_bin = find_bin(z_value, z_edges, n_z_bins);
    const std::int64_t angle_bin = find_bin(angle, angle_edges, n_angle_bins);
    if (z_bin < 0 || angle_bin < 0) {
        return;
    }
    const std::size_t offset =
        (static_cast<std::size_t>(slot) * n_z_bins + static_cast<std::size_t>(z_bin)) * n_angle_bins +
        static_cast<std::size_t>(angle_bin);
    histograms[offset] += 1.0;
}

}  // namespace

extern "C" int waterint_accumulate_oh_orientation(
    const double* oxygen_positions,
    std::size_t n_oxygen,
    const double* hydrogen_positions,
    std::size_t n_hydrogen,
    const std::int64_t* neighbor_counts,
    const std::int64_t* neighbor_matrix,
    std::size_t neighbor_capacity,
    int vector_mode,
    int axis,
    double axis_sign,
    double reference,
    double angle_axis_sign,
    const double* z_edges,
    std::size_t n_z_bins,
    const double* angle_edges,
    std::size_t n_angle_bins,
    double* histograms,
    std::int64_t* bond_counts,
    std::int64_t* sample_counts
) {
    if (
        oxygen_positions == nullptr ||
        hydrogen_positions == nullptr ||
        neighbor_counts == nullptr ||
        neighbor_matrix == nullptr ||
        z_edges == nullptr ||
        angle_edges == nullptr ||
        histograms == nullptr ||
        bond_counts == nullptr ||
        sample_counts == nullptr ||
        neighbor_capacity == 0 ||
        n_z_bins == 0 ||
        n_angle_bins == 0 ||
        axis < 0 ||
        axis > 2 ||
        (vector_mode != 0 && vector_mode != 1)
    ) {
        return 1;
    }

    for (std::size_t i = 0; i < n_oxygen; ++i) {
        const std::int64_t hydrogen_count = neighbor_counts[i];
        if (hydrogen_count < 0 || static_cast<std::size_t>(hydrogen_count) > neighbor_capacity) {
            return 2;
        }
        const int slot = species_slot(hydrogen_count);
        if (slot < 0) {
            continue;
        }

        const double ox = oxygen_positions[3 * i + 0];
        const double oy = oxygen_positions[3 * i + 1];
        const double oz = oxygen_positions[3 * i + 2];
        const double oxygen_axis = oxygen_positions[3 * i + static_cast<std::size_t>(axis)];
        const double z_value = axis_sign * (oxygen_axis - reference);
        const std::int64_t* row = neighbor_matrix + i * neighbor_capacity;

        if (vector_mode == 0) {
            bond_counts[slot] += hydrogen_count;
            for (std::int64_t j = 0; j < hydrogen_count; ++j) {
                const std::int64_t hydrogen_index = row[j];
                if (hydrogen_index < 0 || static_cast<std::size_t>(hydrogen_index) >= n_hydrogen) {
                    return 3;
                }
                const double dx = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 0] - ox;
                const double dy = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 1] - oy;
                const double dz = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 2] - oz;
                const double norm2 = dx * dx + dy * dy + dz * dz;
                if (!(norm2 > 0.0)) {
                    continue;
                }
                const double norm = std::sqrt(norm2);
                const double axis_component = axis == 0 ? dx : (axis == 1 ? dy : dz);
                double cos_theta = angle_axis_sign * axis_component / norm;
                cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
                const double angle = std::acos(cos_theta) * RAD_TO_DEG;
                sample_counts[slot] += 1;
                add_histogram_sample(
                    slot,
                    z_value,
                    angle,
                    z_edges,
                    n_z_bins,
                    angle_edges,
                    n_angle_bins,
                    histograms
                );
            }
            continue;
        }

        double direction[3] = {0.0, 0.0, 0.0};
        std::int64_t valid_bonds = 0;
        for (std::int64_t j = 0; j < hydrogen_count; ++j) {
            const std::int64_t hydrogen_index = row[j];
            if (hydrogen_index < 0 || static_cast<std::size_t>(hydrogen_index) >= n_hydrogen) {
                return 3;
            }
            const double dx = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 0] - ox;
            const double dy = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 1] - oy;
            const double dz = hydrogen_positions[3 * static_cast<std::size_t>(hydrogen_index) + 2] - oz;
            const double norm2 = dx * dx + dy * dy + dz * dz;
            if (!(norm2 > 0.0)) {
                continue;
            }
            const double inverse_norm = 1.0 / std::sqrt(norm2);
            direction[0] += dx * inverse_norm;
            direction[1] += dy * inverse_norm;
            direction[2] += dz * inverse_norm;
            valid_bonds += 1;
        }
        bond_counts[slot] += valid_bonds;
        if (valid_bonds == 0) {
            continue;
        }
        const double direction_norm2 =
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
        if (!(direction_norm2 > 0.0)) {
            continue;
        }
        const double direction_norm = std::sqrt(direction_norm2);
        double cos_theta = angle_axis_sign * direction[static_cast<std::size_t>(axis)] / direction_norm;
        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
        const double angle = std::acos(cos_theta) * RAD_TO_DEG;
        sample_counts[slot] += 1;
        add_histogram_sample(
            slot,
            z_value,
            angle,
            z_edges,
            n_z_bins,
            angle_edges,
            n_angle_bins,
            histograms
        );
    }

    return 0;
}
