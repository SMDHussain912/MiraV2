#include "microphone.hpp"

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <pipewire/pipewire.h>
#include <pipewire/keys.h>
#include <spa/param/audio/format-utils.h>
#include <spa/utils/result.h>
#include <spa/utils/dict.h>
#include <spa/pod/builder.h>
#include <spa/buffer/buffer.h>
#include <thread>

constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;
constexpr size_t CAPTURE_SAMPLES = SAMPLE_RATE * 3;

static void on_global(
    void* data,
    uint32_t id,
    uint32_t permissions,
    const char* type,
    uint32_t version,
    const struct spa_dict* props)
{
    if (std::string(type) != "PipeWire:Interface:Node")
        return;

    if (!props)
        return;

    const char* media_class =
        spa_dict_lookup(props, PW_KEY_MEDIA_CLASS);

    if (!media_class)
        return;

    if (std::string(media_class) != "Audio/Source")
        return;

    Microphone* microphone =
        static_cast<Microphone*>(data);

    microphone->set_target_node(id);
}

static const pw_registry_events registry_events = {
    PW_VERSION_REGISTRY_EVENTS,
    .global = on_global,
    .global_remove = nullptr
};

static void on_sync_done(
    void* data,
    uint32_t id,
    int seq)
{
    Microphone* microphone =
        static_cast<Microphone*>(data);

    try
    {
        microphone->bind_target_node();
        microphone->create_stream();
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "PipeWire initialization error: "
            << error.what()
            << '\n';

        microphone->stop();
    }
}

static const pw_core_events core_events = {
    PW_VERSION_CORE_EVENTS,
    .done = on_sync_done
};

static const pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .destroy = nullptr,
    .state_changed = nullptr,
    .control_info = nullptr,
    .io_changed = nullptr,
    .param_changed = nullptr,
    .add_buffer = nullptr,
    .remove_buffer = nullptr,
    .process = Microphone::on_process
};

Microphone::Microphone()
    : main_loop(nullptr),
      context(nullptr),
      core(nullptr),
      registry(nullptr),
      stream(nullptr),
      node(nullptr),
      registry_listener(),
      core_listener(),
      stream_listener(),
      target_node_id(0),
      node_bound(false),
      recording(false)
{
    pw_init(nullptr, nullptr);

    main_loop = pw_main_loop_new(nullptr);

    context = pw_context_new(
        pw_main_loop_get_loop(main_loop),
        nullptr,
        0
    );

    core = pw_context_connect(
        context,
        nullptr,
        0
    );

    if (!core)
    {
        throw std::runtime_error(
            "Failed to connect to PipeWire"
        );
    }

    registry = pw_core_get_registry(
        core,
        PW_VERSION_REGISTRY,
        0
    );

    if (!registry)
    {
        throw std::runtime_error(
            "Failed to get PipeWire Registry"
        );
    }
    pw_registry_add_listener(
    	registry,
    	&registry_listener,
    	&registry_events,
    	this
    );
    pw_core_add_listener(
        core,
        &core_listener,
        &core_events,
        this
    );
    pw_core_sync(
    	core,
    	PW_ID_CORE,
    	0
    );
}

void Microphone::start()
{
    recording = true;
    audio_samples.clear();

    std::thread(
        [this]()
        {
            pw_main_loop_run(main_loop);
        }
    ).detach();
    while (!node_bound)
    {
        std::this_thread::yield();
    }
}

std::vector<float> Microphone::capture()
{
    pw_main_loop_run(main_loop);

    return audio_samples;
}

void Microphone::set_target_node(uint32_t id)
{
    target_node_id = id;
}

void Microphone::bind_target_node()
{
    if (target_node_id == 0)
    {
        throw std::runtime_error(
            "No target audio node found"
        );
    }

    node = static_cast<pw_node*>(
        pw_registry_bind(
            registry,
            target_node_id,
            PW_TYPE_INTERFACE_Node,
            PW_VERSION_NODE,
            0
        )
    );

    if (!node)
    {
        throw std::runtime_error(
            "Failed to bind target audio node"
        );
    }

    node_bound=true;
}

void Microphone::stop()
{
    recording = false;
}

const std::vector<float>& Microphone::get_audio() const
{
    return audio_samples;
}

void Microphone::create_stream()
{
    if (!node)
    {
        throw std::runtime_error(
            "Target node is not bound"
        );
    }

    pw_properties* properties =
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Communication",
            nullptr
        );

    if (!properties)
    {
        throw std::runtime_error(
            "Failed to create stream properties"
        );
    }

    stream = pw_stream_new(
        core,
        "miraV2 Microphone",
        properties
    );

    if (!stream)
    {
        throw std::runtime_error(
            "Failed to create PipeWire stream"
        );
    }
    pw_stream_add_listener(
        stream,
        &stream_listener,
        &stream_events,
        this
    );

    uint8_t buffer[1024];

    spa_pod_builder builder =
        SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    spa_audio_info_raw audio_info =
        SPA_AUDIO_INFO_RAW_INIT(
            .format = SPA_AUDIO_FORMAT_F32,
            .rate = SAMPLE_RATE,
            .channels = CHANNELS
    );

    const spa_pod* params =
        spa_format_audio_raw_build(
            &builder,
            SPA_PARAM_EnumFormat,
            &audio_info
    );

    int result =
        pw_stream_connect(
            stream,
            PW_DIRECTION_INPUT,
            target_node_id,
            static_cast<pw_stream_flags>(
                PW_STREAM_FLAG_AUTOCONNECT |
                PW_STREAM_FLAG_MAP_BUFFERS
            ),
            &params,
            1
    );

    if (result < 0)
    {
        throw std::runtime_error(
            "Failed to connect PipeWire stream"
        );
    }

}

void Microphone::on_process(void* data)
{
    Microphone* microphone =
        static_cast<Microphone*>(data);

    if (!microphone->stream)
        return;

    pw_buffer* buffer =
        pw_stream_dequeue_buffer(
            microphone->stream
        );

    if (!buffer)
        return;

    if (buffer->buffer->n_datas == 0)
    {
        pw_stream_queue_buffer(
            microphone->stream,
            buffer
        );

        return;
    }

    spa_data* data_ptr =
        &buffer->buffer->datas[0];

    if (!data_ptr->data)
    {
        pw_stream_queue_buffer(
            microphone->stream,
            buffer
        );

        return;
    }

    float* samples = static_cast<float*>(data_ptr->data);
    size_t sample_count = data_ptr->chunk->size / data_ptr->chunk->stride;
    if(microphone->recording)
    {
        microphone->audio_samples.insert(
            microphone->audio_samples.end(),
            samples,
            samples + sample_count
       );
    }
    if (microphone->recording && microphone->audio_samples.size() >= CAPTURE_SAMPLES)
    {
        microphone->audio_samples.resize(CAPTURE_SAMPLES);
    }

    pw_stream_queue_buffer(
        microphone->stream,
        buffer
    );
    if (microphone->recording && microphone->audio_samples.size() >= CAPTURE_SAMPLES)
    {
       microphone->stop();
    }
}

Microphone::~Microphone()
{
}
