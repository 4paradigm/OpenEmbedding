#include <gflags/gflags.h>
#include <pico-core/FileSystem.h>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {

class LabelEncoder {
public:
    LabelEncoder(): _encoder(-1) {}
    size_t encode(int64_t key) {
        if (key == -1) {
            key = std::numeric_limits<int64_t>::max();
        }
        return _encoder.try_emplace(key, _encoder.size()).first->second;
    }

    size_t unique_count() {
        return _encoder.size();
    }
private:
    EasyHashMap<int64_t, size_t> _encoder;
};

class TSVProcesser {
public:
    TSVProcesser(size_t dense_features, size_t sparse_features, size_t repeat)
        : _dense_features(dense_features), _sparse_features(sparse_features), _repeat(repeat),
          _encoders(sparse_features), _key_labels(sparse_features),
          _buffer(64 * (sparse_features + dense_features + 1)), _out_buffer(_buffer.size()) {}

    size_t process(FILE* in, FILE* out) {
        fgets(_buffer.data(), _buffer.size(), in);
        size_t n = strlen(_buffer.data());
        if (n == 0) {
            return 0;
        }
        size_t i = 0;
        for (size_t k = 0; k < _dense_features + 1; ++k) {
            skip_dense(_buffer.data(), i);
            ++i;
        }
        size_t sparse_start = i;
        memcpy(_out_buffer.data(), _buffer.data(), sparse_start);
        for (size_t k = 0; k < _sparse_features; ++k) {
            uint64_t key = parse_sparse(_buffer.data(), i);
            _key_labels[k] = _encoders[k].encode(key);
            ++i;
        }
        for (size_t row = 0; row < _repeat; ++row) {
            size_t i = sparse_start;
            for (size_t k = 0; k < _sparse_features; ++k) {
                output_sparse(_out_buffer.data(), i, _key_labels[k] * _repeat + row);
                _out_buffer[i] = k == _sparse_features - 1 ? '\0' : '\t';
                ++i;
            }
            fprintf(out, "%s\n", _out_buffer.data());
        }
        return _repeat;
    }
    
    void skip_dense(char* buffer, size_t& i) {
        while (buffer[i] && buffer[i] != '\t') ++i;
    }

    int64_t parse_sparse(char* buffer, size_t& i) {
        if (buffer[i] == '\0' || buffer[i] == '\t') {
            return -1;
        }
        int64_t result = 0;
        while (buffer[i] && buffer[i] != '\t') {
            int val = buffer[i] <= '9' && buffer[i] >= '0' ? buffer[i] - '0' : buffer[i] - 'a';
            result = result * 16 + val;
            ++i;
        }
        return result;
    }

    void output_sparse(char* buffer, size_t& i, size_t key) {
        size_t p = i;
        do {
            buffer[i] = '0' + key % 10;
            key /= 10;
            ++i;
        } while (key != 0);
        std::reverse(buffer + p, buffer + i);
    }

    size_t unique_count(size_t sparse_feature) {
        return _encoders[sparse_feature].unique_count() * _repeat;
    }
private:
    size_t _dense_features = 0;
    size_t _sparse_features = 0;
    size_t _repeat = 0;

    std::vector<LabelEncoder> _encoders;
    std::vector<size_t> _key_labels;

    std::vector<char> _buffer;
    std::vector<char> _out_buffer;
    
};


void process(std::string input_dir, std::string output_dir, size_t file_lines, size_t repeat) {
    int day = 1;
    size_t lines = 0;
    auto fout = core::ShellUtility::open(output_dir + "/day_" + std::to_string(day), "w");
    TSVProcesser processer(13, 26, repeat);
    for (std::string input_file: FileSystem::get_file_list(input_dir, "")) {
        SLOG(INFO) << input_file;
        auto fin = core::ShellUtility::open(input_file, "r");
        while (!feof(fin.get())) {
            if (lines >= file_lines) {
                SLOG(INFO) << "day_" << day << " generated";
                ++day;
                lines = 0;
                fout = core::ShellUtility::open(output_dir + "/day_" + std::to_string(day), "w");
            }
            lines += processer.process(fin.get(), fout.get());
        }
    }
    SLOG(INFO) << "day_" << day << " generated";

    fout = core::ShellUtility::open(output_dir + "/meta", "w");
    for (int i = 0; i < 26; ++i) {
        std::string str = pico_lexical_cast<std::string>(processer.unique_count(i));
        fprintf(fout.get(), "C%d %s\n", i + 1, str.c_str());
    }
};

}
} // namespace paradigm4

DEFINE_string(output, "", "");
DEFINE_string(input, "", "");
DEFINE_int32(file_lines, 10000000, "");
DEFINE_int32(repeat, 2, "");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, false);
    paradigm4::pico::process(FLAGS_input, FLAGS_output, FLAGS_file_lines, FLAGS_repeat);
    return 0;
}