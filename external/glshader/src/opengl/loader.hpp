#pragma once

#include <cinttypes>

namespace glshader::process::impl::loader
{
    void* load_function(const char* name) noexcept;
    bool valid() noexcept;
    void reload() noexcept;
}