// Streaming LAMMPS dump relabel I/O (WaterInt v0.7, io/writers/lammpstrj_relabel.py).
// Reads frames of one or more dumps (orthogonal or tilted) (id type x y z ...), skips repeated timesteps (first token of the
// TIMESTEP line), hands id/type/position/box to Python, and writes the frame back with only the type column replaced:
// every other token and header line is copied verbatim, so the output differs from the input only in the types.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

thread_local std::string g_error;

struct Stream {
    std::vector<std::string> paths;
    std::size_t next_path = 0;
    std::ifstream in;
    std::ofstream out;
    std::unordered_set<std::string> seen;
    long long duplicates = 0;
    std::vector<std::string> header;                 // TIMESTEP .. ATOMS header lines of the current frame
    std::vector<std::vector<std::string>> rows;
    int col_id = -1, col_type = -1, col_x = -1, col_y = -1, col_z = -1;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string tok;
    while (is >> tok) out.push_back(tok);
    return out;
}

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

bool open_next(Stream& st) {
    while (st.next_path < st.paths.size()) {
        st.in.close();
        st.in.clear();
        st.in.open(st.paths[st.next_path++]);
        if (!st.in) { g_error = "cannot open " + st.paths[st.next_path - 1]; return false; }
        return true;
    }
    return false;
}

// 1 = frame read, 0 = end of input, -1 = error
int read_frame(Stream& st) {
    std::string line;
    for (;;) {
        bool found = false;
        while (std::getline(st.in, line)) {
            if (starts(line, "ITEM: TIMESTEP")) { found = true; break; }
        }
        if (!found) {
            if (!g_error.empty()) return -1;
            if (!open_next(st)) return g_error.empty() ? 0 : -1;
            continue;
        }
        st.header.assign(1, line);
        std::string step_line, nat_header, nat_line, box_header, b[3], atoms_header;
        if (!std::getline(st.in, step_line) || !std::getline(st.in, nat_header) || !std::getline(st.in, nat_line) ||
            !std::getline(st.in, box_header) || !std::getline(st.in, b[0]) || !std::getline(st.in, b[1]) ||
            !std::getline(st.in, b[2]) || !std::getline(st.in, atoms_header)) { g_error = "truncated frame header"; return -1; }
        if (!starts(nat_header, "ITEM: NUMBER OF ATOMS") || !starts(box_header, "ITEM: BOX BOUNDS") || !starts(atoms_header, "ITEM: ATOMS")) {
            g_error = "unexpected LAMMPS dump layout"; return -1;
        }
        const long long natoms = std::stoll(nat_line);
        st.header.insert(st.header.end(), {step_line, nat_header, nat_line, box_header, b[0], b[1], b[2], atoms_header});
        const std::vector<std::string> cols = split(atoms_header);
        st.col_id = st.col_type = st.col_x = st.col_y = st.col_z = -1;
        for (std::size_t i = 2; i < cols.size(); ++i) {
            const std::string& c = cols[i];
            const int k = static_cast<int>(i) - 2;
            if (c == "id") st.col_id = k; else if (c == "type") st.col_type = k;
            else if (c == "x" || c == "xu") st.col_x = k; else if (c == "y" || c == "yu") st.col_y = k;
            else if (c == "z" || c == "zu") st.col_z = k;
        }
        if (st.col_id < 0 || st.col_type < 0 || st.col_x < 0 || st.col_y < 0 || st.col_z < 0) {
            g_error = "dump needs id type x y z columns"; return -1;
        }
        st.rows.resize(static_cast<std::size_t>(natoms));
        for (long long a = 0; a < natoms; ++a) {
            if (!std::getline(st.in, line)) { g_error = "truncated atom rows"; return -1; }
            st.rows[a] = split(line);
        }
        const std::vector<std::string> step_tok = split(step_line);
        const std::string key = step_tok.empty() ? step_line : step_tok[0];
        if (!st.seen.insert(key).second) { ++st.duplicates; continue; }
        return 1;
    }
}

}  // namespace

extern "C" {

const char* waterint_lmps_error() { return g_error.c_str(); }

void* waterint_lmps_open(const char* const* paths, int n_paths, const char* out_path) {
    g_error.clear();
    auto* st = new Stream();
    for (int i = 0; i < n_paths; ++i) st->paths.emplace_back(paths[i]);
    if (!open_next(*st)) { if (g_error.empty()) g_error = "no input files"; delete st; return nullptr; }
    if (out_path != nullptr && out_path[0] != '\0') {
        st->out.open(out_path);
        if (!st->out) { g_error = std::string("cannot open output ") + out_path; delete st; return nullptr; }
    }
    return st;
}

// Reads the next (non-duplicate) frame into arrays of `capacity` atoms.
// Returns 1 (frame), 0 (end), -1 (error), -2 (frame has more atoms than `capacity`; natoms_out is set).
int waterint_lmps_next(void* handle, long long capacity, long long* natoms_out, long long* step, long long* ids,
                       int* types, double* pos, double* bounds6, double* tilt3) {
    auto* st = static_cast<Stream*>(handle);
    const int r = read_frame(*st);
    if (r <= 0) return r;
    const long long n = static_cast<long long>(st->rows.size());
    *natoms_out = n;
    if (n > capacity) { g_error = "atom count changed between frames"; return -2; }
    *step = std::stoll(split(st->header[1])[0]);
    for (int k = 0; k < 3; ++k) {          // raw BOX BOUNDS values; tilt factors xy, xz, yz (0 for orthogonal boxes)
        const std::vector<std::string> t = split(st->header[5 + k]);
        bounds6[2 * k] = std::stod(t[0]);
        bounds6[2 * k + 1] = std::stod(t[1]);
        tilt3[k] = t.size() > 2 ? std::stod(t[2]) : 0.0;
    }
    for (long long a = 0; a < n; ++a) {
        const std::vector<std::string>& r = st->rows[a];
        ids[a] = std::stoll(r[st->col_id]);
        types[a] = std::stoi(r[st->col_type]);
        pos[3 * a] = std::stod(r[st->col_x]);
        pos[3 * a + 1] = std::stod(r[st->col_y]);
        pos[3 * a + 2] = std::stod(r[st->col_z]);
    }
    return 1;
}

int waterint_lmps_write(void* handle, const int* new_types) {
    auto* st = static_cast<Stream*>(handle);
    if (!st->out) { g_error = "stream has no output"; return -1; }
    for (const std::string& h : st->header) st->out << h << '\n';
    for (std::size_t a = 0; a < st->rows.size(); ++a) {
        std::vector<std::string>& r = st->rows[a];
        r[st->col_type] = std::to_string(new_types[a]);
        for (std::size_t i = 0; i < r.size(); ++i) {
            if (i) st->out << ' ';
            st->out << r[i];
        }
        st->out << '\n';
    }
    return st->out ? 0 : -1;
}

long long waterint_lmps_duplicates(void* handle) { return static_cast<Stream*>(handle)->duplicates; }

void waterint_lmps_close(void* handle) { delete static_cast<Stream*>(handle); }

}  // extern "C"
