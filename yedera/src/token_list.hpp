#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<std::string> parse_token_list(const std::string & raw_value, const std::string & key) {
    std::vector<std::string> tokens;
    std::string current;
    char quote = '\0';

    for (const char character : raw_value) {
        if (quote != '\0') {
            if (character == quote) {
                quote = '\0';
            } else {
                current += character;
            }
            continue;
        }

        if (character == '"' || character == '\'') {
            quote = character;
            continue;
        }

        if (character == ',' || std::isspace(static_cast<unsigned char>(character))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }

        current += character;
    }

    if (quote != '\0') {
        throw std::runtime_error("unterminated quoted value for " + key);
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}
