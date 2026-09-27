// Standalone smoke program for the tokenizer.
//
// Superseded by tests/test_tokenizer.cpp, which runs the same behaviour through
// CTest as part of the build. This file is kept compiling for the manual
// workflow it was originally written for and is consolidated into the test suite
// in ROADMAP Phase 2.
//
// Manual build (from the repository root):
//   g++ -std=c++17 -I. tokenizer/test_tokenizer.cpp tokenizer/tokenizer.cpp \
//       core/text_utils.cpp -o /tmp/test_tokenizer && /tmp/test_tokenizer

#include "tokenizer/tokenizer.hpp"

#include <iostream>

int main()
{
    mira::Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("Open Firefox", mira::Intent::OpenApplication);

    std::cout << "Value: " << result.value << '\n';
    std::cout << "Valid: " << result.valid << '\n';

    return 0;
}
