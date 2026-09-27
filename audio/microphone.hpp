#pragma once

#include <vector>
#include <cstdint>
#include <spa/utils/hook.h>

struct pw_main_loop;
struct pw_stream;
struct pw_context;
struct pw_core;
struct pw_registry;
struct spa_hook;
struct pw_node;

class Microphone
{
public:
    Microphone();
    ~Microphone();

    std::vector<float> capture();
    void set_target_node(uint32_t id);
    void bind_target_node();
    void stop();
    void start();
    void create_stream();
    static void on_process(void* data);
    const std::vector<float>& get_audio() const;

private:
	pw_main_loop* main_loop;
	pw_context* context;
	pw_stream* stream;
	pw_core* core;
	pw_registry* registry;
	pw_node* node;
	struct spa_hook registry_listener;
	struct spa_hook core_listener;
	uint32_t target_node_id;
	bool node_bound;
	bool recording;
	struct spa_hook stream_listener;
	std::vector<float> audio_samples;
};
