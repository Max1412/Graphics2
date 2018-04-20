#include "skip.hpp"

#include "classify.hpp"
#include "control.hpp"

#include <cstring>
#include <sstream>

namespace glshader::process::impl::skip
{
    const char* space(const char* c)
    {
        while (classify::is_space(c) && !classify::is_eof(c)) ++c;
        return c;
    }

    const char* space_rev(const char* c)
    {
        while (classify::is_space(c)) --c;
        return c;
    }

    const char* to_next_space(const char* c)
    {
        using namespace classify;
        while (!is_space(c) && !is_newline(c) && !is_eof(c))
            ++c;
        return c;
    }

    const char* to_next_space(const char* c, char alt)
    {
        using namespace classify;
        while (!is_space(c) && !is_newline(c) && !is_eof(c) && *c != alt)
            ++c;
        return c;
    }

    const char* to_endline(const char* c)
    {
        using namespace classify;
        while (!is_newline(c) && !is_eof(c))
            ++c;
        return c;
    }

    const char* to_next_token(const char* c)
    {
        return space(to_next_space(c));
    }

    const char* over_comments(const char* text_ptr, const files::path& file, int& line, processed_file& processed, std::stringstream& result)
    {
        if (strncmp(text_ptr, "//", 2) == 0)
        {
            while (!classify::is_newline(text_ptr) && classify::is_eof(text_ptr))
                ++text_ptr;
        }
        else if (strncmp(text_ptr, "/*", 2) == 0)
        {
            while (strncmp(text_ptr, "*/", 2) != 0)
            {
                if (classify::is_newline(text_ptr))
                    control::increment_line(line, processed);
                ++text_ptr;
            }

            text_ptr += 2;
            if (processed.version != -1)
                result << control::line_directive(file, line);
        }
        return text_ptr;
    }
}