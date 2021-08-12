#include <gflags/gflags.h>
#include <pico-core/pico_log.h>
#include <pico-core/Master.h>


int main(int argc, char* argv[]) {
    google::InstallFailureSignalHandler();
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, false);

    paradigm4::pico::core::LogReporter::set_id("MASTER", 0);
    paradigm4::pico::core::Master master("");
    
    master.initialize();
    master.finalize();
    return 0;
}
