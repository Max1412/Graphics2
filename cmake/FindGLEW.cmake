#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
# 

set(OpenGL_ROOT_ENV $ENV{OpenGL_ROOT})

IF (WIN32)
	FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		${OpenGL_ROOT_ENV}/include/
	)

    FIND_LIBRARY( GLEW_LIBRARY
        NAMES glew32s.lib
        PATHS ${OpenGL_ROOT_ENV}/lib
    )
    

ELSEIF(APPLE)
	FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
	  PATHS $ENV{OpenGL_ROOT}/include/ /usr/include /usr/local/include /opt/local/include
	)
	 
	FIND_LIBRARY( GLEW_LIBRARY
	  NAMES libGLEW.a
	  PATHS $ENV{OpenGL_ROOT}/lib /usr/lib /usr/local/lib /opt/local/lib
	)

ELSE()
	FIND_PATH(GLEW_INCLUDE_PATH GL/glew.h)
	FIND_LIBRARY(GLEW_LIBRARY
	NAMES GLEW glew32 glew glew32s PATH_SUFFIXES lib64)
ENDIF()
    

SET(GLEW_FOUND "NO")
IF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
	SET(GLEW_LIBRARIES ${GLEW_LIBRARY})
	SET(GLEW_FOUND "YES")
    message("EXTERNAL LIBRARY 'GLEW' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'GLEW' NOT FOUND")
ENDIF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
