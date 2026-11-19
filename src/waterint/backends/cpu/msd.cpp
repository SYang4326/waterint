#include <cmath>
#include <cstddef>
#include <cstdint>

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

void unwrap_delta(double* delta, const double* basis, const std::uint8_t* pbc) {
    double inverse[9];
    if (!invert_3x3(basis, inverse)) return;
    double fractional[3] = {
        delta[0] * inverse[0] + delta[1] * inverse[3] + delta[2] * inverse[6],
        delta[0] * inverse[1] + delta[1] * inverse[4] + delta[2] * inverse[7],
        delta[0] * inverse[2] + delta[1] * inverse[5] + delta[2] * inverse[8],
    };
    for (int axis = 0; axis < 3; ++axis) if (pbc[axis]) fractional[axis] -= std::nearbyint(fractional[axis]);
    delta[0] = fractional[0] * basis[0] + fractional[1] * basis[3] + fractional[2] * basis[6];
    delta[1] = fractional[0] * basis[1] + fractional[1] * basis[4] + fractional[2] * basis[7];
    delta[2] = fractional[0] * basis[2] + fractional[1] * basis[5] + fractional[2] * basis[8];
}

}  // namespace

extern "C" int waterint_msd_sums(
    const double* positions, std::size_t n_frames, std::size_t n_atoms,
    const double* cell_vectors, const std::uint8_t* pbc,
    std::size_t max_lag, std::size_t origin_stride, int dimensions, int excluded_axis,
    double* sums, std::int64_t* samples
) {
    if (positions == nullptr || cell_vectors == nullptr || pbc == nullptr || sums == nullptr || samples == nullptr ||
        n_frames < 2 || n_atoms == 0 || max_lag >= n_frames || origin_stride == 0 || (dimensions != 2 && dimensions != 3) ||
        excluded_axis < 0 || excluded_axis > 2) return 1;
    const std::size_t n_values = n_frames * n_atoms * 3;
    double* unwrapped = new double[n_values];
    for (std::size_t atom = 0; atom < n_atoms; ++atom) {
        for (int axis = 0; axis < 3; ++axis) unwrapped[3 * atom + axis] = positions[3 * atom + axis];
    }
    for (std::size_t frame = 1; frame < n_frames; ++frame) {
        const double* basis = cell_vectors + 9 * frame;
        for (std::size_t atom = 0; atom < n_atoms; ++atom) {
            const std::size_t current = (frame * n_atoms + atom) * 3;
            const std::size_t previous = ((frame - 1) * n_atoms + atom) * 3;
            double delta[3] = {positions[current] - positions[previous], positions[current + 1] - positions[previous + 1], positions[current + 2] - positions[previous + 2]};
            unwrap_delta(delta, basis, pbc);
            unwrapped[current] = unwrapped[previous] + delta[0];
            unwrapped[current + 1] = unwrapped[previous + 1] + delta[1];
            unwrapped[current + 2] = unwrapped[previous + 2] + delta[2];
        }
    }
    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        std::size_t origin_count = 0;
        for (std::size_t frame = 0; frame + lag < n_frames; frame += origin_stride) {
            for (std::size_t atom = 0; atom < n_atoms; ++atom) {
                const std::size_t a = (frame * n_atoms + atom) * 3;
                const std::size_t b = ((frame + lag) * n_atoms + atom) * 3;
                for (int axis = 0; axis < 3; ++axis) {
                    if (dimensions == 2 && axis == excluded_axis) continue;
                    const double delta = unwrapped[b + axis] - unwrapped[a + axis];
                    sum += delta * delta;
                }
            }
            origin_count += 1;
        }
        sums[lag] = sum;
        samples[lag] = static_cast<std::int64_t>(origin_count * n_atoms);
    }
    delete[] unwrapped;
    return 0;
}
