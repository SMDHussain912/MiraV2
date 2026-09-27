#include "whisper.hpp"
#include "whisper_lib/include/whisper.h"
#include <stdexcept>

Whisper::Whisper(const std::string& model_path)
{
	whisper_context_params params = whisper_context_default_params();
	context = whisper_init_from_file_with_params(model_path.c_str(), params);
	if(!context)
	{
		throw std::runtime_error("Failed to load whisper Model");
	}
}

Whisper::~Whisper()
{
    if (context)
    {
        whisper_free(context);
    }
}

std::string Whisper::transcribe(const std::vector<float>& audio)
{
	whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	params.language = "en";
	params.n_threads = 4;
	params.print_progress = false;
	int result = whisper_full(context, params, audio.data(), audio.size());
	if(result != 0)
	{
		throw std::runtime_error("Whisper transcription failed");
	}
	int segment_count = whisper_full_n_segments(context);
	std::string text;
	for(int i = 0; i<segment_count; i++)
	{
		const char* segment = whisper_full_get_segment_text(context, i);
		text += segment;
	}
	return text;
}
