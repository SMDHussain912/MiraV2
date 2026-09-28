// MiraV2 entry point (ROADMAP Phase 8): wires Whisper -> TAMEV -> Tokenizer ->
// Executor through mira::Pipeline. The legacy CommandManager adapter and the
// Phase 5 manual-entry scaffolding are gone: classification owns intent.
//
// Modes:
//   miraV2 --text "Open Firefox"   one utterance, no microphone (CI-friendly)
//   miraV2 --stdin                 read utterances line by line from stdin
//   miraV2                         live mode: hotkey -> microphone -> Whisper
//
// Every run prints the full decision chain (transcript -> intent + probabilities
// -> extracted token -> resolution -> outcome) followed by the user-facing
// response. A below-threshold classification prints a clarification and acts on
// nothing; the next hotkey press is the "did I hear that right?" re-try.

#include "audio/microphone.hpp"
#include "core/paths.hpp"
#include "input/hotkey.hpp"
#include "pipeline/pipeline.hpp"
#include "speech/whisper.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace
{

void print_usage(std::ostream& out)
{
    out << "usage: miraV2 [--text \"<utterance>\"] [--stdin] [--help]\n"
           "  --text <utterance>  classify and execute one utterance (no audio)\n"
           "  --stdin             read one utterance per line from stdin\n"
           "  (no option)         live mode: hold the hotkey, speak, release\n";
}

// Runs one utterance through the pipeline and prints chain + response.
// Returns 0 when the outcome is Ok, 1 otherwise (so shell scripts can assert).
int run_once(mira::Pipeline& pipeline, const std::string& utterance)
{
    mira::PipelineResult result = pipeline.run(utterance);
    std::cout << result.log();
    // User-facing response (the string a future TTS would speak).
    std::cout << "response: " << result.result.message() << '\n';
    return result.result.is_ok() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    bool text_mode = false;
    bool stdin_mode = false;
    std::string text;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--text" && i + 1 < argc)
        {
            text_mode = true;
            text = argv[++i];
        }
        else if (arg.rfind("--text=", 0) == 0)
        {
            text_mode = true;
            text = arg.substr(std::string("--text=").size());
        }
        else if (arg == "--stdin")
        {
            stdin_mode = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(std::cout);
            return 0;
        }
        else
        {
            std::cerr << "unknown argument: " << arg << '\n';
            print_usage(std::cerr);
            return 2;
        }
    }

    mira::Pipeline pipeline;
    const mira::Result loaded = pipeline.load();
    if (!loaded.is_ok())
    {
        std::cerr << "pipeline unavailable: " << loaded.to_string() << '\n';
        return 1;
    }

    if (text_mode)
    {
        return run_once(pipeline, text);
    }

    if (stdin_mode)
    {
        int exit_code = 0;
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }
            exit_code |= run_once(pipeline, line);
        }
        return exit_code;
    }

    // Live mode: hold-to-talk hotkey around a Whisper transcription, then the
    // same pipeline as the text modes. Loops forever; after a clarification
    // ("I didn't catch that") the next hotkey press is the user's retry.
    Hotkey hotkey;
    Microphone microphone;
    Whisper whisper(mira::paths::resolve_data_path("models/ggml-base.en.bin"));

    for (;;)
    {
        hotkey.wait_for_press();
        microphone.start();
        hotkey.wait_for_release();
        microphone.stop();

        const std::vector<float>& audio = microphone.get_audio();
        const std::string utterance = whisper.transcribe(audio);
        std::cout << "You said: " << utterance << '\n';

        run_once(pipeline, utterance);
    }
}
