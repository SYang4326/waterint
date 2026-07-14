#include "../chemistry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct Segment {
    std::int64_t oxygen = -1;
    std::vector<double> mu;
    std::vector<double> stretch;
};

double minimum_image_delta(double delta, double length, bool enabled) {
    return enabled ? delta - std::nearbyint(delta / length) * length : delta;
}

bool parse_pbc(
    const double* cell,
    const std::uint8_t* pbc,
    bool* use_pbc,
    double* cell_lengths
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        use_pbc[axis] = pbc != nullptr && pbc[axis] != 0;
        cell_lengths[axis] = cell == nullptr ? 0.0 : cell[axis];
        if (use_pbc[axis] && cell_lengths[axis] <= 0.0) {
            return false;
        }
    }
    return true;
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

void fft(std::vector<std::complex<double>>* values, bool inverse) {
    std::vector<std::complex<double>>& data = *values;
    const std::size_t size = data.size();
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
    const double pi = std::acos(-1.0);
    for (std::size_t length = 2; length <= size; length <<= 1) {
        const double angle = 2.0 * pi / static_cast<double>(length) * (inverse ? 1.0 : -1.0);
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < size; start += length) {
            std::complex<double> factor(1.0, 0.0);
            for (std::size_t offset = 0; offset < length / 2; ++offset) {
                const std::complex<double> left = data[start + offset];
                const std::complex<double> right = data[start + offset + length / 2] * factor;
                data[start + offset] = left + right;
                data[start + offset + length / 2] = left - right;
                factor *= root;
            }
        }
    }
    if (inverse) {
        for (std::complex<double>& value : data) {
            value /= static_cast<double>(size);
        }
    }
}

// Compute one directional correlation, or the sum of both directions, while
// reusing the same two forward FFTs for symmetrized ssVVCF.
std::vector<double> cross_correlation_fft(
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::size_t max_lag,
    bool symmetric_sum
) {
    const std::size_t length = x.size();
    const std::size_t fft_size = next_power_of_two(2 * length);
    std::vector<std::complex<double>> left(fft_size, 0.0);
    std::vector<std::complex<double>> right(fft_size, 0.0);
    for (std::size_t index = 0; index < length; ++index) {
        left[index] = x[index];
        right[index] = y[index];
    }
    fft(&left, false);
    fft(&right, false);
    for (std::size_t index = 0; index < fft_size; ++index) {
        const std::complex<double> left_spectrum = left[index];
        const std::complex<double> right_spectrum = right[index];
        left[index] = std::conj(left_spectrum) * right_spectrum;
        if (symmetric_sum) {
            left[index] += std::conj(right_spectrum) * left_spectrum;
        }
    }
    fft(&left, true);
    const std::size_t lag_limit = std::min(max_lag, length - 1);
    std::vector<double> correlation(lag_limit + 1, 0.0);
    for (std::size_t lag = 0; lag <= lag_limit; ++lag) {
        correlation[lag] = left[lag].real();
    }
    return correlation;
}

double ramp_sin01(double value) {
    if (value <= 0.0) {
        return 0.0;
    }
    if (value >= 1.0) {
        return 1.0;
    }
    return std::sin(0.5 * std::acos(-1.0) * value);
}

double slab_interface_decay(double z, double z1, double z2, double ramp, bool flip) {
    if (z2 < z1) {
        std::swap(z1, z2);
    }
    const double width = z2 - z1;
    if (width <= 0.0) {
        return flip ? (z > z1 ? 1.0 : 0.0) : (z <= z1 ? 1.0 : 0.0);
    }
    ramp = std::min(std::max(ramp, 0.0), width);
    if (!flip) {
        if (z <= z1) return 1.0;
        if (z >= z2) return 0.0;
        if (ramp <= 0.0 || z <= z2 - ramp) return 1.0;
        return ramp_sin01((z2 - z) / ramp);
    }
    if (z <= z1) return 0.0;
    if (ramp <= 0.0 || z >= z1 + ramp) return 1.0;
    return ramp_sin01((z - z1) / ramp);
}

double top_hat_ramp(double z, double z1, double z2, double ramp) {
    if (z2 < z1) {
        std::swap(z1, z2);
    }
    if (z < z1 || z > z2) return 0.0;
    if (ramp <= 0.0) return 1.0;
    ramp = std::min(ramp, 0.5 * (z2 - z1));
    if (z < z1 + ramp) return ramp_sin01((z - z1) / ramp);
    if (z > z2 - ramp) return ramp_sin01((z2 - z) / ramp);
    return 1.0;
}

double window_factor(double z, bool enabled, int mode, double z1, double z2, double ramp, bool flip) {
    if (!enabled) {
        return 1.0;
    }
    return mode == 1
        ? slab_interface_decay(z, z1, z2, ramp, flip)
        : top_hat_ramp(z, z1, z2, ramp);
}

void atom_velocity(
    const double* positions,
    std::size_t n_frames,
    std::size_t n_atoms,
    std::size_t frame,
    std::size_t atom,
    double dt_ps,
    const double* cell_lengths,
    const bool* use_pbc,
    double* velocity
) {
    const std::size_t left_frame = frame == 0 ? 0 : frame - 1;
    const std::size_t right_frame = frame + 1 >= n_frames ? n_frames - 1 : frame + 1;
    const double denominator = frame == 0 || frame + 1 >= n_frames ? dt_ps : 2.0 * dt_ps;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::size_t left = (left_frame * n_atoms + atom) * 3 + axis;
        const std::size_t right = (right_frame * n_atoms + atom) * 3 + axis;
        const double delta = minimum_image_delta(
            positions[right] - positions[left], cell_lengths[axis], use_pbc[axis]
        );
        velocity[axis] = delta / denominator;
    }
}

double oh_distance2(
    const double* oxygen_positions,
    std::size_t oxygen,
    const double* hydrogen_positions,
    std::size_t hydrogen,
    const double* cell_lengths,
    const bool* use_pbc
) {
    double distance2 = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double delta = minimum_image_delta(
            hydrogen_positions[3 * hydrogen + axis] - oxygen_positions[3 * oxygen + axis],
            cell_lengths[axis],
            use_pbc[axis]
        );
        distance2 += delta * delta;
    }
    return distance2;
}

void accumulate_segment(
    Segment* segment,
    std::size_t max_lag,
    bool symmetrize,
    double* sums,
    std::int64_t* counts
) {
    const std::size_t length = segment->mu.size();
    if (length < 2) {
        segment->mu.clear();
        segment->stretch.clear();
        segment->oxygen = -1;
        return;
    }
    const std::size_t lag_limit = std::min(max_lag, length - 1);
    const std::vector<double> correlation = cross_correlation_fft(
        segment->mu, segment->stretch, lag_limit, symmetrize
    );
    for (std::size_t lag = 0; lag <= lag_limit; ++lag) {
        const std::int64_t samples = static_cast<std::int64_t>(length - lag);
        if (symmetrize) {
            sums[lag] += correlation[lag] * 2.0;
            counts[lag] += 2 * samples;
        } else {
            sums[lag] += correlation[lag];
            counts[lag] += samples;
        }
    }
    segment->mu.clear();
    segment->stretch.clear();
    segment->oxygen = -1;
}

}  // namespace

// Compute the trajectory-mode ssVVCF while Python retains config, zref, FT, and output handling.
extern "C" int waterint_sfg_ssvvcf(
    const double* positions,
    const double* supplied_velocities,
    std::size_t n_frames,
    std::size_t n_atoms,
    const std::int64_t* oxygen_indices,
    std::size_t n_oxygen,
    const std::int64_t* hydrogen_indices,
    std::size_t n_hydrogen,
    const double* zrefs,
    double dt_ps,
    std::size_t max_lag,
    double oh_cutoff,
    const double* cell,
    const std::uint8_t* pbc,
    int mu_mode,
    int symmetrize,
    int flip_sign,
    int duplicate_policy,
    int window_enabled,
    int window_mode,
    double window_z1,
    double window_z2,
    double window_ramp,
    int window_flip,
    double* sums,
    std::int64_t* counts,
    double* stage_seconds
) {
    if (
        positions == nullptr || oxygen_indices == nullptr || hydrogen_indices == nullptr || zrefs == nullptr ||
        sums == nullptr || counts == nullptr || stage_seconds == nullptr || n_frames < 3 || n_atoms == 0 || n_oxygen == 0 ||
        n_hydrogen == 0 || dt_ps <= 0.0 || oh_cutoff <= 0.0 || (mu_mode != 0 && mu_mode != 1) ||
        (duplicate_policy != 0 && duplicate_policy != 1) ||
        (window_enabled != 0 && window_mode != 1 && window_mode != 2)
    ) {
        return 1;
    }
    bool use_pbc[3];
    double cell_lengths[3];
    if (!parse_pbc(cell, pbc, use_pbc, cell_lengths)) {
        return 2;
    }
    for (std::size_t index = 0; index <= max_lag; ++index) {
        sums[index] = 0.0;
        counts[index] = 0;
    }

    std::vector<Segment> active_segments(n_hydrogen);
    std::vector<std::int64_t> active_oxygen(n_hydrogen, -1);
    std::vector<std::int64_t> assigned_oxygen(n_hydrogen, -1);
    std::vector<double> assigned_distance2(n_hydrogen, std::numeric_limits<double>::infinity());
    std::vector<std::int64_t> assignment_count(n_hydrogen, 0);
    std::vector<double> oxygen_positions(3 * n_oxygen);
    std::vector<double> hydrogen_positions(3 * n_hydrogen);
    std::vector<double> oxygen_velocities(3 * n_oxygen);
    std::vector<double> hydrogen_velocities(3 * n_hydrogen);
    stage_seconds[0] = 0.0;
    stage_seconds[1] = 0.0;
    stage_seconds[2] = 0.0;
    stage_seconds[3] = 0.0;
    const auto finalize_segment_timed = [&](Segment* segment) {
        const auto start = std::chrono::steady_clock::now();
        accumulate_segment(segment, max_lag, symmetrize != 0, sums, counts);
        const auto end = std::chrono::steady_clock::now();
        stage_seconds[3] += std::chrono::duration<double>(end - start).count();
    };

    for (std::size_t frame = 0; frame < n_frames; ++frame) {
        const auto assignment_start = std::chrono::steady_clock::now();
        for (std::size_t local = 0; local < n_oxygen; ++local) {
            const std::int64_t global = oxygen_indices[local];
            if (global < 0 || static_cast<std::size_t>(global) >= n_atoms) return 3;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                oxygen_positions[3 * local + axis] = positions[(frame * n_atoms + global) * 3 + axis];
            }
        }
        for (std::size_t local = 0; local < n_hydrogen; ++local) {
            const std::int64_t global = hydrogen_indices[local];
            if (global < 0 || static_cast<std::size_t>(global) >= n_atoms) return 3;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                hydrogen_positions[3 * local + axis] = positions[(frame * n_atoms + global) * 3 + axis];
            }
            assigned_oxygen[local] = -1;
            assigned_distance2[local] = std::numeric_limits<double>::infinity();
            assignment_count[local] = 0;
        }

        // Reuse the package-wide O-H cell list, including the configured minimum-image convention.
        const waterint::CutoffNeighborSearch search(
            oxygen_positions.data(), n_oxygen, hydrogen_positions.data(), n_hydrogen,
            oh_cutoff, cell_lengths, use_pbc
        );
        std::vector<std::vector<std::int64_t>> neighbors(n_oxygen);
        std::vector<std::int64_t> oxygen_h_counts(n_oxygen, 0);
        for (std::size_t oxygen = 0; oxygen < n_oxygen; ++oxygen) {
            const std::vector<std::size_t> local_neighbors = search.collect_indices(oxygen);
            oxygen_h_counts[oxygen] = static_cast<std::int64_t>(local_neighbors.size());
            neighbors[oxygen].reserve(local_neighbors.size());
            for (std::size_t hydrogen : local_neighbors) {
                neighbors[oxygen].push_back(static_cast<std::int64_t>(hydrogen));
            }
        }

        // Match Python's species-grouped traversal before resolving duplicate H assignments.
        for (std::size_t species_slot = 0; species_slot < 5; ++species_slot) {
            for (std::size_t oxygen = 0; oxygen < n_oxygen; ++oxygen) {
                const std::int64_t h_count = oxygen_h_counts[oxygen];
                const std::size_t slot = h_count >= 0 && h_count <= 3
                    ? static_cast<std::size_t>(h_count)
                    : 4;
                if (slot != species_slot) continue;
                for (std::int64_t raw_hydrogen : neighbors[oxygen]) {
                    const std::size_t hydrogen = static_cast<std::size_t>(raw_hydrogen);
                    assignment_count[hydrogen] += 1;
                    const double distance2 = oh_distance2(
                        oxygen_positions.data(), oxygen, hydrogen_positions.data(), hydrogen,
                        cell_lengths, use_pbc
                    );
                    if (assigned_oxygen[hydrogen] < 0 || distance2 < assigned_distance2[hydrogen]) {
                        assigned_oxygen[hydrogen] = static_cast<std::int64_t>(oxygen);
                        assigned_distance2[hydrogen] = distance2;
                    }
                }
            }
        }
        if (duplicate_policy == 1) {
            for (std::size_t hydrogen = 0; hydrogen < n_hydrogen; ++hydrogen) {
                if (assignment_count[hydrogen] > 1) return 4;
            }
        }
        const auto assignment_end = std::chrono::steady_clock::now();
        stage_seconds[0] += std::chrono::duration<double>(assignment_end - assignment_start).count();

        const auto velocity_start = std::chrono::steady_clock::now();
        for (std::size_t oxygen = 0; oxygen < n_oxygen; ++oxygen) {
            const std::size_t atom = static_cast<std::size_t>(oxygen_indices[oxygen]);
            if (supplied_velocities != nullptr) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    oxygen_velocities[3 * oxygen + axis] = supplied_velocities[(frame * n_atoms + atom) * 3 + axis];
                }
            } else {
                atom_velocity(
                    positions, n_frames, n_atoms, frame, atom, dt_ps, cell_lengths, use_pbc,
                    oxygen_velocities.data() + 3 * oxygen
                );
            }
        }
        for (std::size_t hydrogen = 0; hydrogen < n_hydrogen; ++hydrogen) {
            const std::size_t atom = static_cast<std::size_t>(hydrogen_indices[hydrogen]);
            if (supplied_velocities != nullptr) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    hydrogen_velocities[3 * hydrogen + axis] = supplied_velocities[(frame * n_atoms + atom) * 3 + axis];
                }
            } else {
                atom_velocity(
                    positions, n_frames, n_atoms, frame, atom, dt_ps, cell_lengths, use_pbc,
                    hydrogen_velocities.data() + 3 * hydrogen
                );
            }
        }
        const auto velocity_end = std::chrono::steady_clock::now();
        stage_seconds[1] += std::chrono::duration<double>(velocity_end - velocity_start).count();

        const auto signal_start = std::chrono::steady_clock::now();
        const double correlation_before_signal = stage_seconds[3];
        for (std::size_t hydrogen = 0; hydrogen < n_hydrogen; ++hydrogen) {
            const std::int64_t oxygen = assigned_oxygen[hydrogen];
            if (oxygen < 0) {
                if (active_oxygen[hydrogen] >= 0) {
                    finalize_segment_timed(&active_segments[hydrogen]);
                    active_oxygen[hydrogen] = -1;
                }
                continue;
            }
            if (active_oxygen[hydrogen] != oxygen) {
                if (active_oxygen[hydrogen] >= 0) {
                    finalize_segment_timed(&active_segments[hydrogen]);
                }
                active_oxygen[hydrogen] = oxygen;
                active_segments[hydrogen].oxygen = oxygen;
            }

            double r_oh[3];
            double norm2 = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                r_oh[axis] = minimum_image_delta(
                    hydrogen_positions[3 * hydrogen + axis] - oxygen_positions[3 * oxygen + axis],
                    cell_lengths[axis], use_pbc[axis]
                );
                norm2 += r_oh[axis] * r_oh[axis];
            }
            if (!(norm2 > 1.0e-24)) continue;
            const double norm = std::sqrt(norm2);
            double stretch = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                stretch += (
                    hydrogen_velocities[3 * hydrogen + axis] - oxygen_velocities[3 * oxygen + axis]
                ) * r_oh[axis];
            }
            stretch /= norm;
            const double zprime = oxygen_positions[3 * oxygen + 2] - zrefs[frame];
            const double factor = window_factor(
                zprime, window_enabled != 0, window_mode, window_z1, window_z2, window_ramp, window_flip != 0
            );
            const double mu_stretch = factor * stretch * r_oh[2] / norm;
            const double mu_full = factor * (
                hydrogen_velocities[3 * hydrogen + 2] - oxygen_velocities[3 * oxygen + 2]
            );
            double mu = mu_mode == 1 ? mu_stretch : mu_full;
            if (flip_sign != 0) mu *= -1.0;
            active_segments[hydrogen].mu.push_back(mu);
            active_segments[hydrogen].stretch.push_back(stretch);
        }
        const auto signal_end = std::chrono::steady_clock::now();
        const double correlation_during_signal = stage_seconds[3] - correlation_before_signal;
        stage_seconds[2] +=
            std::chrono::duration<double>(signal_end - signal_start).count() -
            correlation_during_signal;
    }

    for (std::size_t hydrogen = 0; hydrogen < n_hydrogen; ++hydrogen) {
        if (active_oxygen[hydrogen] >= 0) {
            finalize_segment_timed(&active_segments[hydrogen]);
        }
    }
    return 0;
}
