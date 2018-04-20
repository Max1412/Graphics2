#include "classify.hpp"
#include "skip.hpp"

#include <cstring>
#include <cctype>

namespace glshader::process::impl::classify
{
    bool is_eof(const char* c)
    {
        return *c=='\0';
    }

    bool is_newline(const char* c)
    {
        return *c=='\n' || *c=='\r';
    }

    bool is_comment(const char* c)
    {
        return strncmp(c, "//", 2) == 0 || strncmp(c, "/*", 2) == 0;
    }

    bool is_space(const char* c)
    {
        return *c==' ' || *c=='\t';
    }

    bool is_name_char(const char* c)
    {
        return isalnum(*c) || *c=='_';
    }

    bool is_directive(const char* c, bool check_before)
    {
        return *c=='#' && (!check_before || is_newline(skip::space_rev(c - 1)));
    }

    bool is_token_equal(const char* c, const char* token, int token_len, bool check_before, bool check_after)
    {
        return (!check_before || !isalpha(*(c - 1))) &&
            (memcmp(c, token, token_len) == 0) &&
            (!check_after || !is_name_char(c + token_len));

    }
}