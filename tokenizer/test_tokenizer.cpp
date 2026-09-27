#include "tokenizer.hpp"

#include <iostream>

int main()
{
    Tokenizer tokenizer;

    TokenResult result =
        tokenizer.process("Open Firefox", "OPEN_APPLICATION");

    std::cout << "Value: " << result.value << '\n';
    std::cout << "Valid: " << result.valid << '\n';

    return 0;
}
