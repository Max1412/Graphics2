#include <glsp/definition.hpp>

#include "preprocessor/skip.hpp"
#include "preprocessor/classify.hpp"

namespace glshader::process
{
    definition_info::definition_info(const char* value)
        : definition_info(std::string(value))
    {

    }

    definition_info::definition_info(const std::string value) : replacement(std::move(value))
    {}

    definition_info::definition_info(const std::vector<std::string> parameters, const std::string replacement) :
        replacement(std::move(replacement)), parameters(std::move(parameters))
    {}

    definition::definition(const std::string& name)
        : name(name)
    {

    }

    definition::definition(const std::string& name, const definition_info& info) 
        : name(name), info(info)
    {

    }

    definition definition::from_format(const std::string& str)
    {
        namespace skip = impl::skip;
        namespace cls = impl::classify;

        const char* begin = impl::skip::space(str.data());
        const char* c = begin;
        while (!cls::is_eof(c) && !cls::is_space(c) && *c != '(')
            ++c;
        const char* end_name = c;
        c = skip::space(c);
        if (cls::is_eof(c))
            return { begin, end_name };
        if (*c == '(')
        {
            definition_info info;
            do
            {
                const char* begin_param = c=skip::space(++c);
                while (!cls::is_eof(c) && !cls::is_space(c) && *c!=',' && *c != ')')
                    ++c;
                const char* end_param = c;

                info.parameters.emplace_back(begin_param, end_param);

                c = skip::space(c);
            } while (!cls::is_eof(c) && *c != ')');
            c = !cls::is_eof(c) ? skip::space(++c) : c;
            info.replacement = std::string{ c, begin + str.size() };
            return definition({ begin, end_name }, info);
        }
        else
        {
            return definition({ begin, end_name }, std::string{ c, begin + str.size() });
        }
    }
}

glsp::definition operator"" _gdef(const char* def, size_t len)
{
    return glsp::definition::from_format({ def, def+len });
}