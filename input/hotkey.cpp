#include "hotkey.hpp"

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <linux/input.h>

constexpr int MIRA_KEY_CTRL  = 29;
constexpr int MIRA_KEY_SHIFT = 42;
constexpr int MIRA_KEY_SUPER = 125;
constexpr int MIRA_KEY_SPACE = 57;

Hotkey::Hotkey()
    : keyboard_fd(-1),
      key_states{}
{
    keyboard_fd = open(
        "/dev/input/event3",
        O_RDONLY
    );

    if (keyboard_fd < 0)
    {
        std::cerr << "Failed to open keyboard device\n";
        return;
    }

    std::cout << "Keyboard device opened successfully!\n";
}

Hotkey::~Hotkey()
{
    if (keyboard_fd >= 0)
    {
        close(keyboard_fd);
    }
}

void Hotkey::wait_for_press()
{
    struct input_event event;

    while (true)
    {
        ssize_t bytes =
            read(
                keyboard_fd,
                &event,
                sizeof(event)
            );

        if (bytes != sizeof(event))
        {
            continue;
        }

        if (event.type != EV_KEY)
        {
            continue;
        }

        if (event.value == 1)
        {
            key_states[event.code] = true;
        }
        else if (event.value == 0)
        {
            key_states[event.code] = false;
        }

        if (is_hotkey_pressed())
        {
            break;
        }
    }
}

void Hotkey::wait_for_release()
{
    struct input_event event;

    while (true)
    {
        ssize_t bytes =
            read(
                keyboard_fd,
                &event,
                sizeof(event)
            );

        if (bytes != sizeof(event))
        {
            continue;
        }

        if (event.type != EV_KEY)
        {
            continue;
        }

        // Ignore key-repeat events.
        if (event.value != 0)
        {
            continue;
        }

        key_states[event.code] = false;

        if (!is_hotkey_pressed())
        {
            break;
        }
    }
}

bool Hotkey::is_pressed(int key)
{
    return key_states[key];
}

bool Hotkey::is_hotkey_pressed()
{
    return key_states[MIRA_KEY_CTRL] && key_states[MIRA_KEY_SHIFT] && key_states[MIRA_KEY_SUPER] && key_states[MIRA_KEY_SPACE];
}
