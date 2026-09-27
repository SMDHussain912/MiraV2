#pragma once

#include <string>

class Hotkey
{
public:
    Hotkey();
    ~Hotkey();

    void wait_for_press();
    void wait_for_release();
    bool is_pressed(int key);
    bool is_hotkey_pressed();

private:
    int keyboard_fd;
    bool key_states[256];
};
