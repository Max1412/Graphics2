cmake_minimum_required(VERSION 3.9)
project(Graphics2)


set(G2_EXECUTABLES_FOLDER       ${PROJECT_SOURCE_DIR}/executables)
set(G2_LIBRARIES_FOLDER			${PROJECT_SOURCE_DIR}/libraries)
set(G2_EXTERNAL_FOLDER			${PROJECT_SOURCE_DIR}/external)
set(G2_BINARIES_FOLDER			${PROJECT_SOURCE_DIR}/bin)

set(G2_RESOURCES_PATH			${PROJECT_SOURCE_DIR}/resources)
set(G2_SHADERS_PATH				${PROJECT_SOURCE_DIR}/shaders)


find_package(GLEW REQUIRED)
find_package(assimp REQUIRED)
find_package(glm 0.9.8 REQUIRED)
find_package(glfw3 3.2.1 REQUIRED)

list(APPEND G2_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_PATH})

add_definitions(
    -DGLFW_INCLUDE_NONE
	-DSTBI_MSC_SECURE_CRT
	-DGLM_FORCE_RADIANS -DGLM_ENABLE_EXPERIMENTAL
    -DNOMINMAX
	-DSOURCE_DIRECTORY="${CMAKE_SOURCE_DIR}/"
	-DRESOURCES_PATH="${G2_RESOURCES_PATH}"
	-DSHADERS_PATH="${G2_SHADERS_PATH}"
)

if(MSVC)
    add_compile_options(/MP /openmp /permissive- /Zc:twoPhase-)
    set(G2_LINKER_FLAGS "/NODEFAULTLIB:libcmt /ignore:4098,4099 /MANIFEST:NO")
    list(APPEND G2_LIBRARIES opengl32)
	file(COPY "${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-vc140-mt.dll" DESTINATION "${G2_BINARIES_FOLDER}/Debug")
	file(COPY "${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-vc140-mt.dll" DESTINATION "${G2_BINARIES_FOLDER}/Release")
else()
endif()


##### internal libraries
list(APPEND G2_INCLUDE_DIRECTORIES "${G2_LIBRARIES_FOLDER}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${G2_BINARIES_FOLDER}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${G2_BINARIES_FOLDER}")
file(GLOB children RELATIVE ${G2_LIBRARIES_FOLDER} ${G2_LIBRARIES_FOLDER}/*)
foreach(subdir ${children})
    if(IS_DIRECTORY ${G2_LIBRARIES_FOLDER}/${subdir})
        file(GLOB_RECURSE G2_SOURCE_FILE "${G2_LIBRARIES_FOLDER}/${subdir}/*.cpp")
        if(NOT G2_SOURCE_FILE)
            # Internal header only library
            set(success 0)
        else()
            add_library(${subdir} ${G2_SOURCE_FILE})
			target_link_libraries(${subdir} PRIVATE glfw GLEW::GLEW ${ASSIMP_LIBRARIES} glm)

            list(APPEND G2_LIBRARIES ${subdir})
            set_target_properties(${subdir} PROPERTIES
											LINKER_LANGUAGE CXX
											CXX_STANDARD 17
											CXX_STANDARD_REQUIRED ON)
            target_include_directories(${subdir} PUBLIC ${G2_INCLUDE_DIRECTORIES})
        endif()
    endif()
endforeach()
include_directories(${G2_INCLUDE_DIRECTORIES})

##### external libraries residing inside the project
list(APPEND G2_INCLUDE_DIRECTORIES ${G2_EXTERNAL_FOLDER})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  "${G2_BINARIES_FOLDER}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${G2_BINARIES_FOLDER}")
file(GLOB children RELATIVE ${G2_EXTERNAL_FOLDER} ${G2_EXTERNAL_FOLDER}/*)
foreach(subdir ${children})
    if(IS_DIRECTORY ${G2_EXTERNAL_FOLDER}/${subdir})
        file(GLOB_RECURSE G2_SOURCE_FILE "${G2_EXTERNAL_FOLDER}/${subdir}/*.cpp")
        if(NOT G2_SOURCE_FILE)
            # Internal header only library
            set(success 0)
        else()
            add_library(${subdir} ${G2_SOURCE_FILE})
            list(APPEND G2_LIBRARIES ${subdir})
			target_link_libraries(${subdir} PRIVATE GLEW::GLEW)
            set_target_properties(${subdir} PROPERTIES
											LINKER_LANGUAGE CXX
											CXX_STANDARD 17
											CXX_STANDARD_REQUIRED ON)
            target_include_directories(${subdir} PUBLIC ${G2_INCLUDE_DIRECTORIES})
        endif()
    endif()
endforeach()
include_directories(${G2_INCLUDE_DIRECTORIES})


##### internal executables
file(GLOB children RELATIVE ${G2_EXECUTABLES_FOLDER} ${G2_EXECUTABLES_FOLDER}/*)
foreach(subdir ${children})
    if(IS_DIRECTORY ${G2_EXECUTABLES_FOLDER}/${subdir})
        file(GLOB_RECURSE EXE_SOURCES "${G2_EXECUTABLES_FOLDER}/${subdir}/*.cpp")

        add_executable(${subdir} ${EXE_SOURCES})
        target_link_libraries(${subdir} PRIVATE ${G2_LIBRARIES})
		target_link_libraries(${subdir} PRIVATE glfw GLEW::GLEW ${ASSIMP_LIBRARIES} glm)
        #file(COPY ${G2_DLLS} DESTINATION "${G2_BINARIES_FOLDER}")
        set_target_properties(${subdir} PROPERTIES
										LINKER_LANGUAGE CXX
										RUNTIME_OUTPUT_DIRECTORY "${G2_BINARIES_FOLDER}"
										CXX_STANDARD 17
										CXX_STANDARD_REQUIRED ON)
        target_include_directories(${subdir} PUBLIC ${G2_INCLUDE_DIRECTORIES})
    endif()
endforeach()
