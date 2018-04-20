#pragma once

#include "files.hpp"
#include <glsp/glsp.hpp>

namespace glshader::process::impl::control
{
    std::string line_directive(const files::path& file, int line);
    void increment_line(int& current_line, processed_file& processed);
}