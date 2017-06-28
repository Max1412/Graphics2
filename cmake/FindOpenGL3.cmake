#
# Try to find OpenGL and include path.
# Once done this will define
#
# OpenGL3_FOUND
# OpenGL3_INCLUDE_PATH
# OpenGL3_LIBRARY
# 

set(OpenGL_ROOT_ENV $ENV{OpenGL_ROOT})

IF (WIN32)
  FIND_PATH( OpenGL3_INCLUDE_PATH 
    NAMES GL/glcorearb.h
    PATHS ${OpenGL_ROOT_ENV}/include/
  )
  SET(OpenGL3_LIBRARY OpenGL32)

ELSEIF (APPLE)
 FIND_PATH(OpenGL3_INCLUDE_PATH OpenGL/gl3.h 
   OpenGL_ROOT_ENV/OpenGL/)
 SET(OpenGL3_LIBRARY "-framework Cocoa -framework OpenGL -framework IOKit" CACHE STRING "OpenGL lib for OSX")
 
ELSE()

SET(OpenGL3_LIBRARY "GL" CACHE STRING "OpenGL lib for Linux")
    FIND_PATH(OpenGL3_INCLUDE_PATH GL/gl.h
      /usr/share/doc/NVIDIA_GLX-1.0/include
      /usr/openwin/share/include
      /opt/graphics/OpenGL/include /usr/X11R6/include
    )
ENDIF ()

SET(OpenGL3_FOUND "NO")
IF (OpenGL3_INCLUDE_PATH)
	SET(OpenGL3_LIBRARIES ${OpenGL3_LIBRARY})
	SET(OpenGL3_FOUND "YES")
    message("EXTERNAL LIBRARY 'OpenGL3' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'OpenGL3' NOT FOUND")
ENDIF (OpenGL3_INCLUDE_PATH)