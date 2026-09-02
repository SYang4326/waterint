#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

bool invert_3x3(const double* matrix, double* inverse) {
    const double a = matrix[0], b = matrix[1], c = matrix[2];
    const double d = matrix[3], e = matrix[4], f = matrix[5];
    const double g = matrix[6], h = matrix[7], i = matrix[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-14) return false;
    const double inv_det = 1.0 / det;
    inverse[0] = (e * i - f * h) * inv_det; inverse[1] = (c * h - b * i) * inv_det; inverse[2] = (b * f - c * e) * inv_det;
    inverse[3] = (f * g - d * i) * inv_det; inverse[4] = (a * i - c * g) * inv_det; inverse[5] = (c * d - a * f) * inv_det;
    inverse[6] = (d * h - e * g) * inv_det; inverse[7] = (b * g - a * h) * inv_det; inverse[8] = (a * e - b * d) * inv_det;
    return true;
}

void fractional_position(const double* position, const double* inverse, double* fractional) {
    fractional[0] = position[0] * inverse[0] + position[1] * inverse[3] + position[2] * inverse[6];
    fractional[1] = position[0] * inverse[1] + position[1] * inverse[4] + position[2] * inverse[7];
    fractional[2] = position[0] * inverse[2] + position[1] * inverse[5] + position[2] * inverse[8];
}

void minimum_image(double* delta, const double* basis, const double* inverse, const std::uint8_t* pbc) {
    double fractional[3];
    fractional_position(delta, inverse, fractional);
    for (int axis = 0; axis < 3; ++axis) if (pbc[axis]) fractional[axis] -= std::nearbyint(fractional[axis]);
    delta[0] = fractional[0] * basis[0] + fractional[1] * basis[3] + fractional[2] * basis[6];
    delta[1] = fractional[0] * basis[1] + fractional[1] * basis[4] + fractional[2] * basis[7];
    delta[2] = fractional[0] * basis[2] + fractional[1] * basis[5] + fractional[2] * basis[8];
}

int wrap_index(int index, int size) {
    const int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

std::size_t flat_index(int x, int y, int z, const int* bins) {
    return static_cast<std::size_t>((z * bins[1] + y) * bins[0] + x);
}

int coordinate_bin(double value, int axis, const double* minima, const double* spans, const int* bins, const std::uint8_t* pbc) {
    const double shifted = pbc[axis] ? value - std::floor(value) : value - minima[axis];
    const int index = static_cast<int>(std::floor(shifted / spans[axis] * bins[axis]));
    return std::max(0, std::min(index, bins[axis] - 1));
}

}  // namespace

extern "C" int waterint_rdf_histogram(
    const double* positions, std::size_t n_atoms,
    const std::int64_t* first_indices, std::size_t n_first,
    const std::int64_t* second_indices, std::size_t n_second,
    int same_selection, double r_max, std::size_t n_bins,
    const double* cell_vectors, const std::uint8_t* pbc, double* counts
) {
    if (positions == nullptr || first_indices == nullptr || second_indices == nullptr || counts == nullptr ||
        r_max <= 0.0 || n_bins == 0) return 1;
    const bool use_pbc = cell_vectors != nullptr && pbc != nullptr && (pbc[0] || pbc[1] || pbc[2]);
    double inverse[9];
    if (use_pbc && !invert_3x3(cell_vectors, inverse)) return 2;

    // Fractional bins preserve the same PBC geometry used in the distance calculation.
    std::vector<std::vector<std::size_t>> neighbor_bins;
    int bins_per_axis[3] = {1, 1, 1};
    double minima[3] = {0.0, 0.0, 0.0};
    double maxima[3] = {1.0, 1.0, 1.0};
    double spans[3] = {1.0, 1.0, 1.0};
    std::vector<double> first_fractional;
    std::vector<double> second_fractional;
    if (use_pbc) {
        first_fractional.resize(n_first * 3);
        second_fractional.resize(n_second * 3);
        for (std::size_t i = 0; i < n_first; ++i) {
            if (first_indices[i] < 0 || static_cast<std::size_t>(first_indices[i]) >= n_atoms) return 3;
            fractional_position(positions + 3 * first_indices[i], inverse, first_fractional.data() + 3 * i);
        }
        for (std::size_t j = 0; j < n_second; ++j) {
            if (second_indices[j] < 0 || static_cast<std::size_t>(second_indices[j]) >= n_atoms) return 3;
            fractional_position(positions + 3 * second_indices[j], inverse, second_fractional.data() + 3 * j);
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!pbc[axis]) {
                minima[axis] = first_fractional[axis];
                maxima[axis] = minima[axis];
                for (std::size_t i = 0; i < n_first; ++i) {
                    minima[axis] = std::min(minima[axis], first_fractional[3 * i + axis]);
                    maxima[axis] = std::max(maxima[axis], first_fractional[3 * i + axis]);
                }
                for (std::size_t j = 0; j < n_second; ++j) {
                    minima[axis] = std::min(minima[axis], second_fractional[3 * j + axis]);
                    maxima[axis] = std::max(maxima[axis], second_fractional[3 * j + axis]);
                }
                spans[axis] = std::max(maxima[axis] - minima[axis], 1e-12);
            }
            const double fractional_reach = r_max * std::sqrt(
                inverse[axis] * inverse[axis] + inverse[3 + axis] * inverse[3 + axis] + inverse[6 + axis] * inverse[6 + axis]
            );
            bins_per_axis[axis] = std::max(1, static_cast<int>(std::floor(spans[axis] / std::max(2.0 * fractional_reach, 1e-12))));
        }
        neighbor_bins.resize(static_cast<std::size_t>(bins_per_axis[0]) * bins_per_axis[1] * bins_per_axis[2]);
        for (std::size_t j = 0; j < n_second; ++j) {
            const double* fractional = second_fractional.data() + 3 * j;
            const int x = coordinate_bin(fractional[0], 0, minima, spans, bins_per_axis, pbc);
            const int y = coordinate_bin(fractional[1], 1, minima, spans, bins_per_axis, pbc);
            const int z = coordinate_bin(fractional[2], 2, minima, spans, bins_per_axis, pbc);
            neighbor_bins[flat_index(x, y, z, bins_per_axis)].push_back(j);
        }
    }

    const double inv_width = static_cast<double>(n_bins) / r_max;
    for (std::size_t i = 0; i < n_first; ++i) {
        if (!use_pbc && (first_indices[i] < 0 || static_cast<std::size_t>(first_indices[i]) >= n_atoms)) return 3;
        std::vector<std::size_t> candidates;
        if (use_pbc) {
            const double* fractional = first_fractional.data() + 3 * i;
            const int base_x = coordinate_bin(fractional[0], 0, minima, spans, bins_per_axis, pbc);
            const int base_y = coordinate_bin(fractional[1], 1, minima, spans, bins_per_axis, pbc);
            const int base_z = coordinate_bin(fractional[2], 2, minima, spans, bins_per_axis, pbc);
            std::vector<std::size_t> visited_bins;
            for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const int x = pbc[0] ? wrap_index(base_x + dx, bins_per_axis[0]) : base_x + dx;
                const int y = pbc[1] ? wrap_index(base_y + dy, bins_per_axis[1]) : base_y + dy;
                const int z = pbc[2] ? wrap_index(base_z + dz, bins_per_axis[2]) : base_z + dz;
                if (x < 0 || x >= bins_per_axis[0] || y < 0 || y >= bins_per_axis[1] || z < 0 || z >= bins_per_axis[2]) continue;
                const std::size_t flat = flat_index(x, y, z, bins_per_axis);
                if (std::find(visited_bins.begin(), visited_bins.end(), flat) != visited_bins.end()) continue;
                visited_bins.push_back(flat);
                candidates.insert(candidates.end(), neighbor_bins[flat].begin(), neighbor_bins[flat].end());
            }
        } else {
            candidates.resize(n_second);
            for (std::size_t j = 0; j < n_second; ++j) candidates[j] = j;
        }
        for (std::size_t j : candidates) {
            if (same_selection && j <= i) continue;
            const std::int64_t first = first_indices[i];
            const std::int64_t second = second_indices[j];
            if (first == second) continue;
            if (!use_pbc && (second < 0 || static_cast<std::size_t>(second) >= n_atoms)) return 3;
            double delta[3] = {positions[3 * second] - positions[3 * first], positions[3 * second + 1] - positions[3 * first + 1], positions[3 * second + 2] - positions[3 * first + 2]};
            if (use_pbc) minimum_image(delta, cell_vectors, inverse, pbc);
            const double distance = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
            if (distance >= r_max) continue;
            const std::size_t bin = static_cast<std::size_t>(distance * inv_width);
            if (bin < n_bins) counts[bin] += 1.0;
        }
    }
    return 0;
}
