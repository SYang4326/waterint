#include <algorithm>
#include <cmath>
#include <cstddef>

extern "C" int waterint_density_histogram_edges(
    const double* values,
    std::size_t n_values,
    const double* bin_edges,
    std::size_t n_bins,
    double* counts
) {
    if (values == nullptr || bin_edges == nullptr || counts == nullptr || n_bins == 0) {
        return 1;
    }

    const double lower = bin_edges[0];
    const double upper = bin_edges[n_bins];
    if (!(upper > lower)) {
        return 2;
    }

    for (std::size_t i = 0; i < n_values; ++i) {
        const double value = values[i];
        if (!std::isfinite(value) || value < lower || value > upper) {
            continue;
        }
        std::size_t bin;
        if (value == upper) {
            bin = n_bins - 1;
        } else {
            const double* first = bin_edges;
            const double* last = bin_edges + n_bins + 1;
            const double* upper_edge = std::upper_bound(first, last, value);
            if (upper_edge == first || upper_edge == last) {
                continue;
            }
            bin = static_cast<std::size_t>((upper_edge - first) - 1);
        }
        counts[bin] += 1.0;
    }

    return 0;
}
