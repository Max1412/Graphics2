#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLM_FOUND
# GLM_INCLUDE_PATH
# 

set(GLM_ROOT_ENV $ENV{GLM_ROOT})

FIND_PATH(GLM_INCLUDE_PATH glm/glm.hpp
	${GLM_ROOT_ENV}
)

SET(GLM_FOUND "NO")
IF (GLM_INCLUDE_PATH)
	SET(GLM_FOUND "YES")
    message("EXTERNAL LIBRARY 'GLM' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'GLM' NOT FOUND")
ENDIF (GLM_INCLUDE_PATH)