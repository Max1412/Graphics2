#include <glsp/compiler.hpp>

#include <glsp/huffman.hpp>
#include "../preprocessor/files.hpp"
#include "../opengl/loader.hpp"
#include "../strings.hpp"
#include <cassert>
#include <fstream>

namespace glshader::process
{
    namespace lgl = impl::loader;

    struct compiled_shader
    {
        uint32_t type;
        std::vector<uint8_t> data;
        std::vector<files::path> dependencies;
        bool success;
    };

    constexpr uint32_t GL_FRAGMENT_SHADER           = 0x8B30;
    constexpr uint32_t GL_VERTEX_SHADER             = 0x8B31;
    constexpr uint32_t GL_GEOMETRY_SHADER           = 0x8DD9;
    constexpr uint32_t GL_TESS_CONTROL_SHADER       = 0x8E88;
    constexpr uint32_t GL_TESS_EVALUATION_SHADER    = 0x8E87;
    constexpr uint32_t GL_COMPUTE_SHADER            = 0x91B9;
    constexpr uint32_t GL_LINK_STATUS               = 0x8B82;
    constexpr uint32_t GL_INFO_LOG_LENGTH           = 0x8B84;
    constexpr uint32_t GL_PROGRAM_BINARY_LENGTH     = 0x8741;

    uint32_t type_from_extension(const files::path& extension)
    {
        if (extension == ".vert")
            return GL_VERTEX_SHADER;
        if (extension == ".frag")
            return GL_FRAGMENT_SHADER;
        if (extension == ".geom")
            return GL_GEOMETRY_SHADER;
        if (extension == ".tesc")
            return GL_TESS_CONTROL_SHADER;
        if (extension == ".tese")
            return GL_TESS_EVALUATION_SHADER;
        if (extension == ".comp")
            return GL_COMPUTE_SHADER;
        return 0;
    }

    compiled_shader load_opengl_binary(const files::path& file, uint32_t type, const std::vector<files::path>& includes, const std::vector<definition>& definitions, const std::string& prefix, const std::string& postfix)
    {
        static uint32_t (*glCreateShaderProgramv)(uint32_t, int, const char**)      = nullptr;
        static void (*glGetProgramiv)(uint32_t, uint32_t, const int*)               = nullptr;
        static void (*glGetProgramInfoLog)(uint32_t, int, int*, char*)              = nullptr;
        static void (*glDeleteProgram)(uint32_t)                                    = nullptr;
        static void (*glGetProgramBinary)(uint32_t, int, int*, uint32_t*, void*)    = nullptr;

        processed_file processed = glsp::preprocess_file(file, includes, definitions);

        // loader should be initialized by glsp::preprocess_file.
        if (lgl::valid() && !(glCreateShaderProgramv && glGetProgramiv && glGetProgramInfoLog && glDeleteProgram && glGetProgramBinary))
        {
            glCreateShaderProgramv  = reinterpret_cast<decltype(glCreateShaderProgramv)>(lgl::load_function("glCreateShaderProgramv"));
            glGetProgramiv          = reinterpret_cast<decltype(glGetProgramiv)>(lgl::load_function("glGetProgramiv"));
            glGetProgramInfoLog     = reinterpret_cast<decltype(glGetProgramInfoLog)>(lgl::load_function("glGetProgramInfoLog"));
            glDeleteProgram         = reinterpret_cast<decltype(glDeleteProgram)>(lgl::load_function("glDeleteProgram"));
            glGetProgramBinary      = reinterpret_cast<decltype(glGetProgramBinary)>(lgl::load_function("glGetProgramBinary"));
        }
        else if(!lgl::valid())
        {
            glCreateShaderProgramv  = nullptr;
            glGetProgramiv          = nullptr;
            glGetProgramInfoLog     = nullptr;
            glDeleteProgram         = nullptr;
            glGetProgramBinary      = nullptr;
        }

        compiled_shader result;

        if (!(glCreateShaderProgramv && glGetProgramiv && glGetProgramInfoLog && glDeleteProgram && glGetProgramBinary))
        {
            result.success = false;
            syntax_error("Function Loader", 0, strings::serr_loader_failed);
            return result;
        }

        const char* sources[3] = {
            prefix.c_str(),
            processed.contents.c_str(),
            postfix.c_str()
        };

        const auto id = glCreateShaderProgramv(type, 3, sources);
        int success = 0; glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success)
        {
            int log_length;
            glGetProgramiv(id, GL_INFO_LOG_LENGTH, &log_length);
            std::string log(log_length, ' ');
            glGetProgramInfoLog(id, log_length, &log_length, log.data());
            glDeleteProgram(id);
            syntax_error("Linking", 0, log);
            result.success = false;
            return result;
        }

        int length; glGetProgramiv(id, GL_PROGRAM_BINARY_LENGTH, &length);
        result.data.resize(length);
        glGetProgramBinary(id, length, &length, &result.type, result.data.data());
        glDeleteProgram(id);
        result.dependencies.push_back(file);
        result.dependencies.insert(result.dependencies.end(), processed.dependencies.begin(), processed.dependencies.end());
        return result;
    }

    compiler::compiler(const std::string& extension, const glsp::files::path& cache_dir)
        : _cache_dir(cache_dir)
    {
        set_extension(extension);
    }

    void compiler::set_extension(const std::string& ext)
    {
        assert(ext.length() > 0);
        _extension = ext[0] == '.' ? ext : ('.' + ext);
    }

    void compiler::set_cache_dir(const glsp::files::path& dir)
    {
        _cache_dir = dir;
    }

    void compiler::set_default_prefix(const std::string& prefix)
    {
        _default_prefix = prefix;
    }

    void compiler::set_default_postfix(const std::string& postfix)
    {
        _default_postfix = postfix;
    }

    struct shader_file_header
    {
        format type;             // SPRV or GBIN.
        uint32_t version;               // version as 100*maj + 10*min + 1*rev.

        uint32_t info_tag;              // INFO
        uint32_t dependencies_length;   // byte size of dependencies
        uint32_t dependencies_count;
        uint32_t binary_format;         // If binary received from opengl, this contains the binary format GLenum
        uint32_t binary_length;         // byte size of binary

        uint32_t data_tag;              // DATA
    };
    
    shader_binary compiler::compile(const glsp::files::path& shader, format format, bool force_reload, std::vector<glsp::files::path> includes, std::vector<glsp::definition> definitions)
    {
        shader_binary result;
        files::path dst = absolute(shader);
        const auto hash = std::hash<std::string>()(dst.string());
        if (!files::exists(_cache_dir))
        {
            files::create_directories(_cache_dir);
        }
        dst = files::path(_cache_dir) / (std::to_string(hash) + _extension);
        uint32_t internal_format = 0;

        bool reload = false;
        if (!force_reload && exists(dst))
        {
            shader_file_header header;
            std::ifstream input(dst, std::ios::binary);
            input.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.type != format || header.info_tag != make_tag("INFO") || header.data_tag != make_tag("DATA"))
                reload = true;

            if (!reload)
            {
                for (int i = 0; i < static_cast<int>(header.dependencies_count); ++i)
                {
                    std::time_t last_write{ 0 };
                    uint32_t slength{ 0 };

                    input.read(reinterpret_cast<char*>(&last_write), sizeof(last_write));
                    input.read(reinterpret_cast<char*>(&slength), sizeof(slength));
                    std::string dep_file;
                    dep_file.resize(slength);
                    input.read(dep_file.data(), slength);

                    // Treat missing dependency as non-needed.
                    if (!files::exists(dep_file))
                        continue;

                    if (std::chrono::system_clock::to_time_t(files::last_write_time(dep_file)) != last_write)
                    {
                        reload = true;
                        break;
                    }
                }
            }
            if (!reload)
            {
                std::basic_string<uint8_t> compressed;
                compressed.resize(header.binary_length);
                input.read(reinterpret_cast<char*>(compressed.data()), header.binary_length);

                result.data = compress::huffman::decode(compressed).to_container<decltype(result.data)>();
                internal_format = header.binary_format;
                result.format = header.binary_format;
            }
        }
        else
        {
            reload = true;
        }

        if (reload)
        {
            shader_file_header header;
            header.type = format;
            header.info_tag = make_tag("INFO");
            header.data_tag = make_tag("DATA");
            std::vector<files::path> dependencies;

            includes.insert(includes.end(), _include_directories.begin(), _include_directories.end());
            definitions.insert(definitions.end(), _definitions.begin(), _definitions.end());
            switch (header.type)
            {
            case format::gl_binary:
            {
                compiled_shader compiled = load_opengl_binary(shader, type_from_extension(shader.extension()), includes, definitions, _default_prefix, _default_postfix);
                result.data = std::move(compiled.data);
                dependencies = std::move(compiled.dependencies);
                internal_format = compiled.type;
                result.format = internal_format;
            } break;
            case format::spirv:
                syntax_error("Loader", 0, strfmt(strings::serr_unsupported, "SPIR-V"));
            default:
                result.data.clear();
                return result;
            }

            std::stringstream buf;
            for (auto dep : dependencies)
            {
                std::time_t t = std::chrono::system_clock::to_time_t(last_write_time(dep));
                auto str = dep.string();
                uint32_t len = static_cast<uint32_t>(str.length());
                buf.write(reinterpret_cast<const char*>(&t), sizeof(t));
                buf.write(reinterpret_cast<const char*>(&len), sizeof(len));
                buf.write(str.data(), str.length());
            }

            auto deps = buf.str();

            const std::vector<uint8_t> compressed =  compress::huffman::encode(result.data).to_container<decltype(compressed)>();
            header.version = 100;
            header.dependencies_length = static_cast<uint32_t>(deps.size());
            header.dependencies_count = static_cast<uint32_t>(dependencies.size());
            header.binary_format = internal_format;
            header.binary_length = static_cast<uint32_t>(compressed.size());

            if (!result.data.empty())
            {
                std::ofstream out(dst, std::ios::binary);
                out.write(reinterpret_cast<const char*>(&header), sizeof(header));
                out.write(deps.data(), deps.size());
                out.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                out.close();
            }
        }

        return result;
    }
}