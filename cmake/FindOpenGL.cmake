#
# Try to find OpenGL and include path.
# Once done this will define
#
# OpenGL3_FOUND
# OpenGL3_INCLUDE_PATH
# OpenGL3_LIBRARY
#

set(OpenGL_DIR ${GLEW_INCLUDE_PATH})

if(SYS_WINDOWS)
    FIND_PATH(OpenGL_INCLUDE_PATH
            NAMES gl/glcorearb.h
            PATHS ${OpenGL_DIR}/include)
		
	set(OpenGL_LIBRARY OpenGL32)
	
elseif(SYS_APPLE)
	FIND_PATH(OpenGL_INCLUDE_PATH OpenGL/gl3.h
            ${OpenGL_DIR}/mac/)
    SET(OpenGL_LIBRARY "-framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo" CACHE STRING "OpenGL lib for OSX")
else()
	SET(OpenGL_LIBRARY "gl" CACHE STRING "OpenGL lib for Linux")
    FIND_PATH(OpenGL_INCLUDE_PATH GL/gl.h
            /usr/share/doc/NVIDIA_GLX-1.0/include
            /usr/openwin/share/include
            /opt/graphics/OpenGL/include /usr/X11R6/include
            )
endif()

SET(OpenGL_FOUND "NO")
IF (OpenGL_INCLUDE_PATH)
	SET(OpenGL_LIBRARIES ${OpenGL_LIBRARY})
	SET(OpenGL_FOUND "YES")
    message("EXTERNAL LIBRARY 'OpenGL' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'OpenGL' NOT FOUND")
ENDIF (OpenGL_INCLUDE_PATH)