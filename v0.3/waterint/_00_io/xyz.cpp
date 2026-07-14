#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

struct WaterintXYZData {
    std::size_t n_frames;
    std::size_t n_atoms;
    double* positions;
    char* symbols;
    std::size_t symbols_size;
    char* error;
};

}

namespace {

void set_error(WaterintXYZData* out, const std::string& message) {
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

const char* skip_space(const char* text) {
    while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') {
        ++text;
    }
    return text;
}

bool parse_size_line(const std::string& line, std::size_t* value) {
    char* end = nullptr;
    errno = 0;
    const char* start = skip_space(line.c_str());
    const unsigned long parsed = std::strtoul(start, &end, 10);
    if (errno != 0 || end == start) {
        return false;
    }
    *value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_atom_line(const std::string& line, std::string* symbol, double* x, double* y, double* z) {
    const char* cursor = skip_space(line.c_str());
    const char* symbol_start = cursor;
    while (*cursor != '\0' && *cursor != ' ' && *cursor != '\t' && *cursor != '\r' && *cursor != '\n') {
        ++cursor;
    }
    if (cursor == symbol_start) {
        return false;
    }
    *symbol = std::string(symbol_start, static_cast<std::size_t>(cursor - symbol_start));

    char* end = nullptr;
    errno = 0;
    cursor = skip_space(cursor);
    *x = std::strtod(cursor, &end);
    if (errno != 0 || end == cursor) {
        return false;
    }
    cursor = skip_space(end);
    *y = std::strtod(cursor, &end);
    if (errno != 0 || end == cursor) {
        return false;
    }
    cursor = skip_space(end);
    *z = std::strtod(cursor, &end);
    if (errno != 0 || end == cursor) {
        return false;
    }
    return true;
}

}  // namespace

extern "C" int waterint_read_xyz_file(
    const char* path,
    std::size_t max_frames,
    WaterintXYZData* out
) {
    if (path == nullptr || out == nullptr) {
        return 1;
    }
    *out = WaterintXYZData{0, 0, nullptr, nullptr, 0, nullptr};

    FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        set_error(out, std::string("Could not open XYZ file: ") + path);
        return 2;
    }

    std::vector<double> positions;
    std::vector<std::string> first_symbols;
    std::string line;
    std::size_t frame_index = 0;
    std::size_t n_atoms = 0;

    while (max_frames == 0 || frame_index < max_frames) {
        if (!read_line(handle, line)) {
            break;
        }
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::size_t frame_atoms = 0;
        if (!parse_size_line(line, &frame_atoms)) {
            std::fclose(handle);
            set_error(out, "Bad XYZ atom-count line.");
            return 3;
        }
        if (frame_atoms == 0) {
            std::fclose(handle);
            set_error(out, "XYZ frames must contain at least one atom.");
            return 4;
        }
        if (frame_index == 0) {
            n_atoms = frame_atoms;
            positions.reserve(n_atoms * 3 * (max_frames == 0 ? 1 : max_frames));
        } else if (frame_atoms != n_atoms) {
            std::fclose(handle);
            set_error(out, "C++ XYZ reader requires a fixed atom count.");
            return 5;
        }

        if (!read_line(handle, line)) {
            std::fclose(handle);
            set_error(out, "Unexpected EOF after XYZ atom-count line.");
            return 6;
        }

        for (std::size_t atom = 0; atom < n_atoms; ++atom) {
            if (!read_line(handle, line)) {
                std::fclose(handle);
                set_error(out, "Unexpected EOF inside XYZ frame.");
                return 7;
            }
            std::string symbol;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (!parse_atom_line(line, &symbol, &x, &y, &z)) {
                std::fclose(handle);
                set_error(out, "Bad XYZ atom row.");
                return 8;
            }
            if (frame_index == 0) {
                first_symbols.push_back(symbol);
            } else if (symbol != first_symbols[atom]) {
                std::fclose(handle);
                set_error(out, "C++ XYZ reader requires fixed atom symbols and ordering.");
                return 12;
            }
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
        }
        ++frame_index;
    }

    std::fclose(handle);

    if (frame_index == 0) {
        set_error(out, "No XYZ frames were found.");
        return 9;
    }

    const std::size_t position_count = positions.size();
    out->positions = static_cast<double*>(std::malloc(position_count * sizeof(double)));
    if (out->positions == nullptr) {
        set_error(out, "Could not allocate XYZ position buffer.");
        return 10;
    }
    std::memcpy(out->positions, positions.data(), position_count * sizeof(double));

    std::string joined_symbols;
    for (const std::string& symbol : first_symbols) {
        joined_symbols += symbol;
        joined_symbols += '\n';
    }
    out->symbols_size = joined_symbols.size();
    out->symbols = static_cast<char*>(std::malloc(out->symbols_size + 1));
    if (out->symbols == nullptr) {
        std::free(out->positions);
        out->positions = nullptr;
        set_error(out, "Could not allocate XYZ symbol buffer.");
        return 11;
    }
    std::memcpy(out->symbols, joined_symbols.c_str(), out->symbols_size + 1);

    out->n_frames = frame_index;
    out->n_atoms = n_atoms;
    return 0;
}

extern "C" void waterint_free_xyz_data(WaterintXYZData* data) {
    if (data == nullptr) {
        return;
    }
    std::free(data->positions);
    std::free(data->symbols);
    std::free(data->error);
    *data = WaterintXYZData{0, 0, nullptr, nullptr, 0, nullptr};
}
