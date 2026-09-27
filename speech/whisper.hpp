#pragma once

#include <string>
#include <vector>
#include "whisper_lib/include/whisper.h"

class Whisper
{
public:
	Whisper(const std::string& model_path);
	~Whisper();
	std::string transcribe(const std::vector<float>& audio);
private:
	whisper_context* context;
};
