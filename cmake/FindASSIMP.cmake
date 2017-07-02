#
# Try to find Assimp library and include path.
# Once done this will define
#
# ASSIMP_FOUND
# ASSIMP_INCLUDE_PATH
# ASSIMP_LIBRARY
# 

set(ASSIMP_ROOT_ENV $ENV{ASSIMP_ROOT})

IF(APPLE)
    set(ASSIMP_ROOT_ENV $ENV{CVK_DEPENDENCIES_OSX}/assimp/)
ENDIF(APPLE)

IF (MINGW)
    FIND_PATH( ASSIMP_INCLUDE_PATH assimp/defs.h
        ${ASSIMP_ROOT_ENV}/include
    )

    FIND_LIBRARY( ASSIMP_LIBRARY
        NAMES assimp
        PATHS ${ASSIMP_ROOT_ENV}/libMinGW
    )
	
	execute_process(COMMAND ${CMAKE_COMMAND}  -E make_directory
		${PROJECT_BINARY_DIR}/bin/
	)
    execute_process(COMMAND ${CMAKE_COMMAND}  -E  copy_if_different
        ${ASSIMP_ROOT_ENV}/libMinGW/libassimp.dll
        ${PROJECT_BINARY_DIR}/bin/
    )

	
ELSEIF (MSVC)
    FIND_PATH( ASSIMP_INCLUDE_PATH assimp/defs.h
        ${ASSIMP_ROOT_ENV}/include
    )

    FIND_LIBRARY( ASSIMP_LIBRARY
        NAMES assimp
        PATHS ${ASSIMP_ROOT_ENV}/lib/assimp_release-dll_win32
    )
	
    foreach (CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
        execute_process(COMMAND ${CMAKE_COMMAND}  -E make_directory
            ${PROJECT_BINARY_DIR}/bin/${CONFIGURATION_TYPE}/
        )
        execute_process(COMMAND ${CMAKE_COMMAND}  -E  copy_if_different
            ${ASSIMP_ROOT_ENV}/lib/assimp_release-dll_win32/Assimp32.dll
            ${PROJECT_BINARY_DIR}/bin/${CONFIGURATION_TYPE}/
        )
    endforeach()

ELSEIF(APPLE)

    FIND_PATH(ASSIMP_INCLUDE_PATH assimp/defs.h
    ${ASSIMP_ROOT_ENV}/include)
    
    FIND_LIBRARY( ASSIMP_LIBRARY
        NAMES libassimp.a libassimp.dylib
        PATHS ${ASSIMP_ROOT_ENV}/lib)
    SET(ASSIMP_LIBRARY z ${ASSIMP_LIBRARY})

ELSE()

    FIND_PATH(ASSIMP_INCLUDE_PATH assimp/defs.h)
    FIND_LIBRARY(ASSIMP_LIBRARY
        NAMES assimp
    PATH_SUFFIXES dynamic)

ENDIF ()



SET(ASSIMP_FOUND "NO")
IF (ASSIMP_INCLUDE_PATH AND ASSIMP_LIBRARY)
	SET(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY})
	SET(ASSIMP_FOUND "YES")
    message("EXTERNAL LIBRARY 'ASSIMP' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'ASSIMP' NOT FOUND")
ENDIF (ASSIMP_INCLUDE_PATH AND ASSIMP_LIBRARY)
