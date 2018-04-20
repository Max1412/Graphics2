#pragma once

#include <set>
#include <string>

namespace glshader::process::impl::ext
{
    void enable_extension(const char* extension);
    bool extension_available(const std::string& extension);
    const std::set<std::string>& extensions() noexcept;
}