/*******************************************************************************/
/* File     definition.hpp
/* Author   Johannes Braun
/* Created  30.03.2018
/*
/* Wrapper for in-code #define directives.
/*******************************************************************************/

#pragma once

#include "config.hpp"
#include <string>
#include <vector>

namespace glshader::process
{
    /* Holds all the needed information for resolving a definition. */
    struct definition_info
    {
        definition_info() = default;
        /* Use the given string value as a macro replacement (no macro parameters). */
        definition_info(std::string value);
        /* Use the given string value as a macro replacement (no macro parameters). */
        definition_info(const char* value);
        /* Use the given value as a macro replacement (no macro parameters). Will be converted to a string via std::to_string.*/
        template<typename T, typename = decltype(std::to_string(std::declval<T>()))>
        definition_info(const T& value) : definition_info(std::to_string(value)) {}
        /* Full initialization with parameters and an according macro replacement. */
        definition_info(std::vector<std::string> parameters, std::string replacement);

        /* The string that will be inserted and resolved when expanding the macro. */
        std::string replacement;
        /* The macro's parameters. */
        std::vector<std::string> parameters;
    };

    struct definition
    {
        definition() = default;
        /* Parameterless and valueless definition with a name. */
        definition(const std::string& name);
        /* Full initialization with name and info. */
        definition(const std::string& name, const definition_info& info);

        /* Parse string and construct definition from it.
        Format: MACRO(p0, p1, ..., pn) replacement */
        static definition from_format(const std::string& str);

        /* Macro name written in code. */
        std::string name;
        /* Macro info like replacement and parameters. */
        definition_info info;
    };
}

/* Calls and returns glsp::definition::from_format({def, def+len}). */
glsp::definition operator"" _gdef(const char* def, size_t len);