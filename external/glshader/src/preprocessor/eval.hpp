#pragma once

#include "files.hpp"

namespace glshader::process::impl::operation
{
    /*** 
    Takes a const char* substring with a given length and tries to evaluate it's value.
    Evaluates arithmetical, logical and comparison operations on integral values.
    ***/
    int eval(const char* x, int len, const files::path& current_file, const int current_line);
}