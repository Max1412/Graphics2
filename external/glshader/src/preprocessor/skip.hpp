#pragma once

#include "files.hpp"
#include <glsp/glsp.hpp>

namespace glshader::process::impl::skip
{
    const char* space           (const char* c);
    const char* space_rev       (const char* c);
    const char* to_next_space   (const char* c);
    const char* to_next_space   (const char* c, char alt);
    const char* to_endline      (const char* c);
    const char* to_next_token   (const char* c);
    const char* over_comments   (const char* text_ptr, const files::path& file, int& line, processed_file& processed, std::stringstream& result);
}