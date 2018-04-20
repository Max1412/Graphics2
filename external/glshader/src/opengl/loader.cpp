#include "loader.hpp"

#include <array>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace glshader::process::impl::loader
{
    class function_loader
    {
    public:
        function_loader()
        {
            for (size_t i = 0; i < std::size(libs); ++i) {
#ifdef _WIN32
                hnd = LoadLibraryA(libs[i]);
#else
                hnd = dlopen(libs[i], RTLD_LAZY | RTLD_GLOBAL);
#endif
                if (hnd != nullptr)
                    break;
            }
        }

        void load_getters() noexcept
        {
#ifdef __APPLE__
            get_fun = nullptr;
            get_ctx_fun = reinterpret_cast<decltype(get_ctx_fun)>(get_handle(hnd, "CGLGetCurrentContext"));
#elif defined _WIN32
            get_fun     = reinterpret_cast<decltype(get_fun)>(get_handle(hnd, "wglGetProcAddress"));
            get_ctx_fun = reinterpret_cast<decltype(get_ctx_fun)>(get_handle(hnd, "wglGetCurrentContext"));
#else
            get_fun     = reinterpret_cast<decltype(get_fun)>(get_handle(hnd, "glXGetProcAddressARB"));
            get_ctx_fun = reinterpret_cast<decltype(get_ctx_fun)>(get_handle(hnd, "glXGetCurrentContext"));
#endif
            ctx = get_ctx_fun();
        }

        bool valid() const
        {
            return ctx && ctx == get_ctx_fun();
        }

        ~function_loader() 
        {
#ifdef _WIN32
            FreeLibrary(HMODULE(hnd));
#else
            dlclose(hnd);
#endif
        }

        void* get(const char* name) const
        {
            void* addr = get_fun ? get_fun(name) : nullptr;
            return addr ? addr : get_handle(hnd, name);
        }

    private:
        void *hnd;
        void* ctx;

        void* get_handle(void* handle, const char* name) const
        {
#if defined _WIN32
            return static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
            return dlsym(handle, name);
#endif
        }

        void* (*get_fun)(const char*) = nullptr;
        void* (*get_ctx_fun)() = nullptr;

#ifdef __APPLE__
        constexpr static std::array<const char *, 4> libs ={
            "../Frameworks/OpenGL.framework/OpenGL",
            "/Library/Frameworks/OpenGL.framework/OpenGL",
            "/System/Library/Frameworks/OpenGL.framework/OpenGL",
            "/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL"
        };
#elif defined _WIN32
        constexpr static std::array<const char *, 2> libs ={ "opengl32.dll" };
#else
#if defined __CYGWIN__
        constexpr static std::array<const char *, 3> libs ={
            "libGL-1.so",
#else
        constexpr static std::array<const char *, 2> libs ={
#endif
            "libGL.so.1",
            "libGL.so"
        };
#endif
    };

    function_loader& get_loader()
    {
        static function_loader l;
        return l;
    }

    bool valid() noexcept 
    {
        return get_loader().valid();
    }

    void reload() noexcept
    {
        get_loader().load_getters();
    }

    void* load_function(const char* name) noexcept
    {
        return get_loader().get(name);
    }
}