/*******************************************************************************/
/* File     compiler.hpp
/* Author   Johannes Braun
/* Created  01.04.2018
/*
/* Wrapper for the proprietary binary format interface in OpenGL to 
/* cache binary versions of loaded shaders for shorter loading times.
/*******************************************************************************/

#pragma once

#include "glsp.hpp"

namespace glshader::process
{
    /* Pack a 4-byte char sequence into a uint32_t. Used in binary file section markers and format tags. */
    constexpr uint32_t make_tag(const char name[4])
    {
        return (name[0] << 0) | (name[1] << 8) | (name[2] << 16) | (name[3] << 24);
    }

    /* The base format of the binary source. */
    enum class format : uint32_t
    {
        gl_binary   = make_tag("GBIN"),     /* Use system's proprietary vendor binary format. */
        spirv       = make_tag("SPRV")      /* Use SPIR-V format. NOT SUPPORTED AT THE MOMENT! */
    };

    /* The resulting binary shader data. */
    struct shader_binary
    {
        uint32_t format;            /* The vendor binary format, used as binaryFormat parameter in glProgramBinary. */
        std::vector<uint8_t> data;  /* The binary data. */
    };

    /* A wrapper class containing state information about compiling shaders.
    Derives from glsp::state and can therefore preprocess and compile shader files.
    Additionally to the state class, you can set file extensions for the cached binary files,
    which will be saved into the cache directory with their filename being their text-format-shader's source path's hash value.
    There is also the option to set a prefix and a postfix for OpenGL shaders. This might be useful if you wish for 
    all shaders to have the same #version and #extension declarations, as well as layout(bindless_<object>) uniform; declarations. */
    class compiler : public glsp::state
    {
    public:
        /* A compiler constructed with this constructor will in it's unchanged state save binaries in the following path:
        <cache_dir>/<shader_path_hash>.<extension> 
        If passed a file extension not starting with a '.', it will be prepended.*/
        compiler(const std::string& extension, const glsp::files::path& cache_dir);

        /* Replace the file extension with which binaries will be saved. */
        void set_extension(const std::string& ext);

        /* Set the directory in which compiled binaries will be saved and from where they will be loaded. */
        void set_cache_dir(const glsp::files::path& dir);

        /* Set a common source code prefix for all compiled shaders. This will NOT be preprocessed! */
        void set_default_prefix(const std::string& prefix);

        /* Set a common source code postfix for all compiled shaders. This will NOT be preprocessed! */
        void set_default_postfix(const std::string& postfix);

        /* Preprocess, compile, save and return binary data of the given shader file. If force_reload is set to false, the binary file already exists
        and the internal time stamp matches the shader's last editing time, the binary file will be loaded and returned directly instead. 
        The parameters "includes" and "definitions" can add special include paths and definitions for this one compilation process. */
        shader_binary compile(const glsp::files::path& shader, format format, bool force_reload = false, std::vector<glsp::files::path> includes ={}, std::vector<glsp::definition> definitions ={});

    private:
        std::string _default_prefix;
        std::string _default_postfix;
        std::string _extension;
        glsp::files::path _cache_dir;
    };
}