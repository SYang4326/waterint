#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

extern "C" {

struct WaterintLammpstrjData {
    std::size_t n_frames;
    std::size_t n_atoms;
    double* positions;
    double* velocities;
    std::int64_t* types;
    double* cells;
    double* cell_vectors;
    double* cell_origins;
    std::uint8_t* triclinic;
    std::int64_t* steps;
    char* error;
};

}

namespace {

void set_error(WaterintLammpstrjData* out, const std::string& message) {
    if (out == nullptr) {
        return;
    }
    char* buffer = static_cast<char*>(std::malloc(message.size() + 1));
    if (buffer == nullptr) {
        return;
    }
    std::memcpy(buffer, message.c_str(), message.size() + 1);
    out->error = buffer;
}

bool read_line(FILE* handle, std::string& line) {
    line.clear();
    char buffer[8192];
    while (true) {
        if (std::fgets(buffer, sizeof(buffer), handle) == nullptr) {
            return !line.empty();
        }
        line += buffer;
        if (!line.empty() && line.back() == '\n') {
            return true;
        }
    }
}

std::string trim_eol(std::string line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

bool starts_with(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool parse_int64(const std::string& line, std::int64_t* value) {
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(line.c_str(), &end, 10);
    if (errno != 0 || end == line.c_str()) {
        return false;
    }
    *value = static_cast<std::int64_t>(parsed);
    return true;
}

bool parse_size(const std::string& line, std::size_t* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(line.c_str(), &end, 10);
    if (errno != 0 || end == line.c_str()) {
        return false;
    }
    *value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_bounds(const std::string& line, double* lo, double* hi) {
    char* end = nullptr;
    errno = 0;
    *lo = std::strtod(line.c_str(), &end);
    if (errno != 0 || end == line.c_str()) {
        return false;
    }
    *hi = std::strtod(end, &end);
    if (errno != 0) {
        return false;
    }
    return true;
}

bool parse_bounds_with_tilt(const std::string& line, double* lo, double* hi, double* tilt) {
    char* end = nullptr;
    errno = 0;
    *lo = std::strtod(line.c_str(), &end);
    if (errno != 0 || end == line.c_str()) {
        return false;
    }
    *hi = std::strtod(end, &end);
    if (errno != 0) {
        return false;
    }
    *tilt = std::strtod(end, &end);
    if (errno != 0) {
        return false;
    }
    return true;
}

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

int column_index(const std::vector<std::string>& columns, const std::string& name) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool is_triclinic_box_header(const std::string& line) {
    const std::vector<std::string> words = split_words(line);
    if (words.size() > 6) {
        return true;
    }
    for (std::size_t i = 3; i < words.size(); ++i) {
        if (words[i] == "xy" || words[i] == "xz" || words[i] == "yz") {
            return true;
        }
    }
    return false;
}

bool parse_atom_row(
    const std::string& line,
    int type_col,
    int x_col,
    int y_col,
    int z_col,
    int vx_col,
    int vy_col,
    int vz_col,
    std::int64_t* type_value,
    double* x,
    double* y,
    double* z,
    double* vx,
    double* vy,
    double* vz
) {
    const int needed = std::max(
        std::max(std::max(type_col, x_col), std::max(y_col, z_col)),
        std::max(vx_col, std::max(vy_col, vz_col))
    );
    if (needed < 0) {
        return false;
    }
    const char* cursor = line.c_str();
    for (int column = 0; column <= needed; ++column) {
        while (*cursor != '\0' && std::isspace(static_cast<unsigned char>(*cursor))) {
            ++cursor;
        }
        if (*cursor == '\0') {
            return false;
        }

        char* end = nullptr;
        if (column == type_col) {
            errno = 0;
            const double parsed = std::strtod(cursor, &end);
            if (errno != 0 || end == cursor) {
                return false;
            }
            *type_value = static_cast<std::int64_t>(parsed);
            cursor = end;
        } else if (column == x_col) {
            errno = 0;
            *x = std::strtod(cursor, &end);
            if (errno != 0 || end == cursor) {
                return false;
            }
            cursor = end;
        } else if (column == y_col) {
            errno = 0;
            *y = std::strtod(cursor, &end);
            if (errno != 0 || end == cursor) {
                return false;
            }
            cursor = end;
        } else if (column == z_col) {
            errno = 0;
            *z = std::strtod(cursor, &end);
            if (errno != 0 || end == cursor) {
                return false;
            }
            cursor = end;
        } else if (column == vx_col || column == vy_col || column == vz_col) {
            errno = 0;
            const double parsed = std::strtod(cursor, &end);
            if (errno != 0 || end == cursor) {
                return false;
            }
            if (column == vx_col) *vx = parsed;
            if (column == vy_col) *vy = parsed;
            if (column == vz_col) *vz = parsed;
            cursor = end;
        } else {
            while (*cursor != '\0' && !std::isspace(static_cast<unsigned char>(*cursor))) {
                ++cursor;
            }
        }
    }
    return true;
}

}  // namespace

extern "C" int waterint_read_lammpstrj_file(
    const char* path,
    std::size_t max_frames,
    int has_start_timestep,
    std::int64_t start_timestep,
    std::size_t stride,
    WaterintLammpstrjData* out
) {
    if (path == nullptr || out == nullptr) {
        return 1;
    }
    if (stride == 0) {
        return 1;
    }
    *out = WaterintLammpstrjData{0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        set_error(out, std::string("Could not open LAMMPS dump file: ") + path);
        return 2;
    }

    std::string line;
    std::size_t n_atoms = 0;
    std::size_t n_frames = 0;
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<std::int64_t> types;
    std::vector<double> cells;
    std::vector<double> cell_vectors;
    std::vector<double> cell_origins;
    std::vector<std::uint8_t> triclinic_flags;
    std::vector<std::int64_t> steps;
    bool velocity_layout_known = false;
    bool trajectory_has_velocities = false;
    std::size_t frame_index = 0;

    while (max_frames == 0 || n_frames < max_frames) {
        if (!read_line(handle, line)) {
            break;
        }
        if (trim_eol(line) != "ITEM: TIMESTEP") {
            continue;
        }

        if (!read_line(handle, line)) {
            std::fclose(handle);
            set_error(out, "Unexpected EOF after ITEM: TIMESTEP.");
            return 3;
        }
        std::int64_t timestep = 0;
        if (!parse_int64(line, &timestep)) {
            std::fclose(handle);
            set_error(out, "Bad timestep row.");
            return 4;
        }

        if (!read_line(handle, line) || trim_eol(line) != "ITEM: NUMBER OF ATOMS") {
            std::fclose(handle);
            set_error(out, "Expected ITEM: NUMBER OF ATOMS.");
            return 5;
        }
        if (!read_line(handle, line)) {
            std::fclose(handle);
            set_error(out, "Unexpected EOF after ITEM: NUMBER OF ATOMS.");
            return 6;
        }
        std::size_t frame_atoms = 0;
        if (!parse_size(line, &frame_atoms)) {
            std::fclose(handle);
            set_error(out, "Bad atom-count row.");
            return 7;
        }
        if (frame_atoms == 0) {
            std::fclose(handle);
            set_error(out, "LAMMPS dump frames must contain at least one atom.");
            return 8;
        }
        if (n_frames == 0) {
            n_atoms = frame_atoms;
            const std::size_t reserve_frames = max_frames == 0 ? 1 : max_frames;
            positions.reserve(reserve_frames * n_atoms * 3);
            velocities.reserve(reserve_frames * n_atoms * 3);
            types.reserve(reserve_frames * n_atoms);
            cells.reserve(reserve_frames * 3);
            cell_vectors.reserve(reserve_frames * 9);
            cell_origins.reserve(reserve_frames * 3);
            triclinic_flags.reserve(reserve_frames);
            steps.reserve(reserve_frames);
        } else if (frame_atoms != n_atoms) {
            std::fclose(handle);
            set_error(out, "C++ LAMMPS dump reader requires a fixed atom count.");
            return 9;
        }

        const bool should_yield =
            (has_start_timestep == 0 || timestep >= start_timestep) &&
            (frame_index % stride == 0);

        if (!read_line(handle, line) || !starts_with(line, "ITEM: BOX BOUNDS")) {
            std::fclose(handle);
            set_error(out, "Expected ITEM: BOX BOUNDS.");
            return 10;
        }
        const bool triclinic = is_triclinic_box_header(line);
        double cell_lengths[3] = {0.0, 0.0, 0.0};
        double origin[3] = {0.0, 0.0, 0.0};
        double vectors[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (triclinic) {
            double xlo_bound = 0.0;
            double xhi_bound = 0.0;
            double ylo_bound = 0.0;
            double yhi_bound = 0.0;
            double zlo_bound = 0.0;
            double zhi_bound = 0.0;
            double xy = 0.0;
            double xz = 0.0;
            double yz = 0.0;
            if (
                !read_line(handle, line) || !parse_bounds_with_tilt(line, &xlo_bound, &xhi_bound, &xy) ||
                !read_line(handle, line) || !parse_bounds_with_tilt(line, &ylo_bound, &yhi_bound, &xz) ||
                !read_line(handle, line) || !parse_bounds_with_tilt(line, &zlo_bound, &zhi_bound, &yz)
            ) {
                std::fclose(handle);
                set_error(out, "Bad triclinic BOX BOUNDS rows.");
                return 12;
            }
            const double xlo = xlo_bound - std::min({0.0, xy, xz, xy + xz});
            const double xhi = xhi_bound - std::max({0.0, xy, xz, xy + xz});
            const double ylo = ylo_bound - std::min(0.0, yz);
            const double yhi = yhi_bound - std::max(0.0, yz);
            const double zlo = zlo_bound;
            const double zhi = zhi_bound;
            cell_lengths[0] = xhi - xlo;
            cell_lengths[1] = yhi - ylo;
            cell_lengths[2] = zhi - zlo;
            if (cell_lengths[0] <= 0.0 || cell_lengths[1] <= 0.0 || cell_lengths[2] <= 0.0) {
                std::fclose(handle);
                set_error(out, "Bad triclinic BOX BOUNDS lengths.");
                return 12;
            }
            origin[0] = xlo;
            origin[1] = ylo;
            origin[2] = zlo;
            vectors[0] = cell_lengths[0];
            vectors[1] = 0.0;
            vectors[2] = 0.0;
            vectors[3] = xy;
            vectors[4] = cell_lengths[1];
            vectors[5] = 0.0;
            vectors[6] = xz;
            vectors[7] = yz;
            vectors[8] = cell_lengths[2];
        } else {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!read_line(handle, line)) {
                    std::fclose(handle);
                    set_error(out, "Unexpected EOF inside BOX BOUNDS.");
                    return 11;
                }
                double lo = 0.0;
                double hi = 0.0;
                if (!parse_bounds(line, &lo, &hi)) {
                    std::fclose(handle);
                    set_error(out, "Bad BOX BOUNDS row.");
                    return 12;
                }
                origin[axis] = lo;
                cell_lengths[axis] = hi - lo;
                if (cell_lengths[axis] <= 0.0) {
                    std::fclose(handle);
                    set_error(out, "Bad BOX BOUNDS length.");
                    return 12;
                }
            }
            vectors[0] = cell_lengths[0];
            vectors[4] = cell_lengths[1];
            vectors[8] = cell_lengths[2];
        }

        if (!read_line(handle, line) || !starts_with(line, "ITEM: ATOMS")) {
            std::fclose(handle);
            set_error(out, "Expected ITEM: ATOMS.");
            return 13;
        }
        std::vector<std::string> header_words = split_words(line);
        if (header_words.size() < 6) {
            std::fclose(handle);
            set_error(out, "Bad ITEM: ATOMS header.");
            return 14;
        }
        std::vector<std::string> columns(header_words.begin() + 2, header_words.end());
        const int type_col = column_index(columns, "type");
        const int x_col = column_index(columns, "x");
        const int y_col = column_index(columns, "y");
        const int z_col = column_index(columns, "z");
        const int vx_col = column_index(columns, "vx");
        const int vy_col = column_index(columns, "vy");
        const int vz_col = column_index(columns, "vz");
        if (type_col < 0 || x_col < 0 || y_col < 0 || z_col < 0) {
            std::fclose(handle);
            set_error(out, "LAMMPS dump missing one of: type, x, y, z.");
            return 15;
        }
        const bool has_velocities = vx_col >= 0 || vy_col >= 0 || vz_col >= 0;
        if (has_velocities && (vx_col < 0 || vy_col < 0 || vz_col < 0)) {
            std::fclose(handle);
            set_error(out, "LAMMPS dump must include vx, vy, and vz together.");
            return 20;
        }
        if (!velocity_layout_known) {
            velocity_layout_known = true;
            trajectory_has_velocities = has_velocities;
        } else if (trajectory_has_velocities != has_velocities) {
            std::fclose(handle);
            set_error(out, "LAMMPS dump velocity columns must be present in every frame or none.");
            return 21;
        }

        if (!should_yield) {
            for (std::size_t atom = 0; atom < frame_atoms; ++atom) {
                if (!read_line(handle, line)) {
                    std::fclose(handle);
                    set_error(out, "Unexpected EOF while skipping atom rows.");
                    return 16;
                }
            }
            ++frame_index;
            continue;
        }

        steps.push_back(timestep);
        cells.push_back(cell_lengths[0]);
        cells.push_back(cell_lengths[1]);
        cells.push_back(cell_lengths[2]);
        for (std::size_t i = 0; i < 9; ++i) {
            cell_vectors.push_back(vectors[i]);
        }
        cell_origins.push_back(origin[0]);
        cell_origins.push_back(origin[1]);
        cell_origins.push_back(origin[2]);
        triclinic_flags.push_back(triclinic ? 1 : 0);
        for (std::size_t atom = 0; atom < n_atoms; ++atom) {
            if (!read_line(handle, line)) {
                std::fclose(handle);
                set_error(out, "Unexpected EOF inside atom rows.");
                return 16;
            }
            std::int64_t type_value = 0;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            if (!parse_atom_row(
                line, type_col, x_col, y_col, z_col, vx_col, vy_col, vz_col,
                &type_value, &x, &y, &z, &vx, &vy, &vz
            )) {
                std::fclose(handle);
                set_error(out, "Bad atom row.");
                return 17;
            }
            types.push_back(type_value);
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
            if (trajectory_has_velocities) {
                velocities.push_back(vx);
                velocities.push_back(vy);
                velocities.push_back(vz);
            }
        }
        ++n_frames;
        ++frame_index;
    }

    std::fclose(handle);

    if (n_frames == 0) {
        set_error(out, "No LAMMPS dump frames were found.");
        return 18;
    }

    out->positions = static_cast<double*>(std::malloc(positions.size() * sizeof(double)));
    out->velocities = velocities.empty()
        ? nullptr
        : static_cast<double*>(std::malloc(velocities.size() * sizeof(double)));
    out->types = static_cast<std::int64_t*>(std::malloc(types.size() * sizeof(std::int64_t)));
    out->cells = static_cast<double*>(std::malloc(cells.size() * sizeof(double)));
    out->cell_vectors = static_cast<double*>(std::malloc(cell_vectors.size() * sizeof(double)));
    out->cell_origins = static_cast<double*>(std::malloc(cell_origins.size() * sizeof(double)));
    out->triclinic = static_cast<std::uint8_t*>(std::malloc(triclinic_flags.size() * sizeof(std::uint8_t)));
    out->steps = static_cast<std::int64_t*>(std::malloc(steps.size() * sizeof(std::int64_t)));
    if (
        out->positions == nullptr || (!velocities.empty() && out->velocities == nullptr) ||
        out->types == nullptr || out->cells == nullptr || out->cell_vectors == nullptr ||
        out->cell_origins == nullptr || out->triclinic == nullptr || out->steps == nullptr
    ) {
        std::free(out->positions);
        std::free(out->velocities);
        std::free(out->types);
        std::free(out->cells);
        std::free(out->cell_vectors);
        std::free(out->cell_origins);
        std::free(out->triclinic);
        std::free(out->steps);
        *out = WaterintLammpstrjData{0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        set_error(out, "Could not allocate LAMMPS dump output buffers.");
        return 19;
    }
    std::memcpy(out->positions, positions.data(), positions.size() * sizeof(double));
    if (!velocities.empty()) {
        std::memcpy(out->velocities, velocities.data(), velocities.size() * sizeof(double));
    }
    std::memcpy(out->types, types.data(), types.size() * sizeof(std::int64_t));
    std::memcpy(out->cells, cells.data(), cells.size() * sizeof(double));
    std::memcpy(out->cell_vectors, cell_vectors.data(), cell_vectors.size() * sizeof(double));
    std::memcpy(out->cell_origins, cell_origins.data(), cell_origins.size() * sizeof(double));
    std::memcpy(out->triclinic, triclinic_flags.data(), triclinic_flags.size() * sizeof(std::uint8_t));
    std::memcpy(out->steps, steps.data(), steps.size() * sizeof(std::int64_t));
    out->n_frames = n_frames;
    out->n_atoms = n_atoms;
    return 0;
}

extern "C" void waterint_free_lammpstrj_data(WaterintLammpstrjData* data) {
    if (data == nullptr) {
        return;
    }
    std::free(data->positions);
    std::free(data->velocities);
    std::free(data->types);
    std::free(data->cells);
    std::free(data->cell_vectors);
    std::free(data->cell_origins);
    std::free(data->triclinic);
    std::free(data->steps);
    std::free(data->error);
    *data = WaterintLammpstrjData{0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
}
