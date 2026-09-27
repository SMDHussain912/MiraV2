#include "speech/whisper.hpp"
#include "audio/microphone.hpp"
#include "input/hotkey.hpp"
#include "cmdmgr/cmdmgr.hpp"
#include <iostream>
#include <vector>

int main()
{
    Hotkey hotkey;
    Microphone microphone;

    hotkey.wait_for_press();

    microphone.start();

    hotkey.wait_for_release();

    microphone.stop();
    const std::vector<float>& audio = microphone.get_audio();
    Whisper whisper("models/ggml-base.en.bin");
    std::string text = whisper.transcribe(audio);

    std::cout << "You said: "<< text<< '\n';
    CommandManager command_manager;

    Command command = command_manager.process(text);

    std::cout << "Action: " << command.action << '\n';
    std::cout << "Valid: " << command.valid << '\n';
    std::cout << "Target: " << command.target << '\n';
    return 0;
}
