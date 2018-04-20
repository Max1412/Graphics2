#pragma once

namespace glshader::process::impl::classify
{
    bool is_eof         (const char* c);
    bool is_newline     (const char* c);
    bool is_comment     (const char* c);
    bool is_space       (const char* c);
    bool is_name_char   (const char* c);
    bool is_directive   (const char* c, bool check_before = true);
    bool is_token_equal (const char* c, const char* token, int token_len, bool check_before = true, bool check_after = true);
}