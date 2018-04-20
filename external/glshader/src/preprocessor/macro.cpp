#include "macro.hpp"

#include "classify.hpp"
#include "control.hpp"
#include "skip.hpp"
#include "extensions.hpp"
#include "../strings.hpp"

#include <sstream>

namespace glshader::process::impl::macro
{
    namespace cls = impl::classify;
    namespace ctrl = impl::control;
    namespace skip = impl::skip;

    bool is_defined(const std::string& val, const processed_file& processed)
    {
        if (std::strncmp(val.data(), "GL_", 3) && ext::extension_available(val))
            return true;
        return processed.definitions.count(val) != 0;
    }

    bool is_macro(const char* text_ptr, processed_file& processed)
    {
        const auto begin = text_ptr;
        while (cls::is_name_char(text_ptr))
            ++text_ptr;

        const std::string str(begin, text_ptr);
        return is_defined(str, processed);
    }

    std::string expand_macro(const std::string& name, const char* param_start, const int param_length,
        const files::path& current_file, const int current_line, processed_file& processed)
    {
        std::stringstream stream;
        if (processed.definitions.count(name) == 0)
            return name;

        auto info = processed.definitions.at(name);

        if (info.parameters.empty())
            return info.replacement;

        std::vector<std::string> inputs;

        if (param_start != nullptr)
        {
            std::stringstream param_stream({ param_start, param_start + param_length });
            std::string in;

            while (std::getline(param_stream, in, ','))
            {
                const auto bg = skip::space(in.data());
                inputs.push_back(in.data() + (bg - in.data()));
            }
        }

        if (inputs.size() != info.parameters.size() || (info.parameters.size() >= inputs.size() - 1 && inputs.back() ==
            "..."))
        {
            syntax_error(current_file, current_line, strfmt(strings::serr_non_matching_argc, name));
            return "";
        }


        bool skip_stream = false;
        for (int replacement_offset = 0; replacement_offset < static_cast<int>(info.replacement.length()); ++
            replacement_offset)
        {
            for (int parameter = 0; parameter < static_cast<int>(info.parameters.size()); ++parameter)
            {
                if (cls::is_token_equal(&info.replacement[replacement_offset], info.parameters[parameter].data(),
                    static_cast<unsigned>(info.parameters[parameter].length()),
                    replacement_offset != 0))
                {
                    skip_stream = true;
                    stream << inputs[parameter];
                    replacement_offset += static_cast<int>(info.parameters[parameter].length() - 1);
                    break;
                }
                if (cls::is_token_equal(&info.replacement[replacement_offset], "__VA_ARGS__", 11, replacement_offset != 0) && info
                    .parameters[parameter] == "...")
                {
                    skip_stream = true;
                    for (auto input_parameter = parameter; input_parameter != inputs.size(); ++input_parameter)
                        stream << inputs[input_parameter];
                    break;
                }
            }
            if (skip_stream)
                skip_stream = false;
            else
                stream << info.replacement[replacement_offset];
        }
        return stream.str();
    }

    std::string expand(const char* text_ptr, const char*& text_ptr_after,
        const files::path& current_file, const int current_line,
        processed_file& processed)
    {
        std::string line(text_ptr, skip::to_endline(text_ptr));
        bool first_replacement = true;
        while (true)
        {
            const auto begin = text_ptr;
            while (cls::is_name_char(text_ptr))
                ++text_ptr;

            const auto begin_params = skip::space(text_ptr);
            auto end_params = begin_params - 1;
            if (*begin_params == '(')
            {
                while (*end_params != ')' && !cls::is_eof(end_params))
                    ++end_params;
            }

            if (!is_macro(begin, processed))
            {
                if (cls::is_eof(text_ptr) || cls::is_newline(text_ptr) || line.empty() || text_ptr == &line[line.size() - 1])
                    break;
                ++text_ptr;
                if (cls::is_eof(text_ptr) || cls::is_newline(text_ptr) || line.empty() || text_ptr == &line[line.size() - 1])
                    break;
                continue;
            }

            const auto params_start = *begin_params == '(' ? begin_params + 1 : nullptr;
            const auto params_length = *begin_params == '(' ? end_params - params_start : 0;

            std::string expanded_macro = expand_macro({ begin, text_ptr }, params_start, static_cast<int>(params_length),
                current_file, current_line, processed);

            if (first_replacement)
            {
                first_replacement = false;
                text_ptr_after = end_params;
                line = expanded_macro;
            }
            else
            {
                line.replace(line.begin() + static_cast<size_t>(begin - line.data()),
                    line.begin() + static_cast<size_t>(text_ptr + params_length + 2 - line.data()), expanded_macro.begin(),
                    expanded_macro.end());
            }

            text_ptr = line.data();
            bool enable_test_macro = true;
            while (!line.empty() && ((!enable_test_macro || !is_macro(text_ptr, processed)) && text_ptr != &line[line.size() - 1]))
            {
                if (!cls::is_name_char(text_ptr))
                    enable_test_macro = true;
                else
                    enable_test_macro = false;
                ++text_ptr;
            }

            if (line.empty() || text_ptr == &line[line.size() - 1])
                break;
        }
        if (line.empty())
            return line;
        if (line[line.size() - 1] == '\0')
            line[line.size() - 1] = '\n';
        return line;
    }
}