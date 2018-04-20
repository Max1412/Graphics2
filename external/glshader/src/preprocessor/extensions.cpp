#include "extensions.hpp"
#include <set>
#include <string>

namespace glshader::process::impl::ext
{
    std::set<std::string> _extensions;
    void enable_extension(const char* extension)
    {
        _extensions.emplace(extension);
    }

    bool extension_available(const std::string& extension)
    {
        return _extensions.count(extension) != 0;
    }

    const std::set<std::string>& extensions() noexcept
    {
        return _extensions;
    }
}