cmake_minimum_required(VERSION 3.20)
project(vkterrain-renderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOMOC_VERBOSE ON)

# ─── Dependencies ─────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
find_package(Vulkan REQUIRED)
find_package(spdlog QUIET)
find_package(nlohmann_json REQUIRED)

# ─── Sources ──────────────────────────────────────────────────────────────────
file(GLOB_RECURSE VKTR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/vktr/*.hpp")
file(GLOB_RECURSE VKTR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME} STATIC ${VKTR_HEADERS} ${VKTR_SOURCES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# VkCompute headers + VMA — path relative to Hesiod repo root
# These are available via the vkcompute target's public includes
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Qt6::Core
    Qt6::Widgets
    Vulkan::Vulkan
    nlohmann_json::nlohmann_json
    vkcompute
)

if(spdlog_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)
endif()

# ─── Compile definitions ─────────────────────────────────────────────────────

# Shader output dir — same as compute shaders for consistency
set(VKTR_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders" CACHE PATH "SPIR-V output dir")

target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    VKTR_SHADER_PATH="${VKTR_SHADER_OUTPUT_DIR}"
  PRIVATE
    $<$<CONFIG:Debug>:VKCOMPUTE_VALIDATION_LAYERS>
)

# MSVC flags
if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

# ─── Shader compilation ──────────────────────────────────────────────────────
set(VKTR_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/vktr/shaders")

file(GLOB VKTR_SHADER_SOURCES
  "${VKTR_SHADER_SOURCE_DIR}/*.vert"
  "${VKTR_SHADER_SOURCE_DIR}/*.frag"
)

set(VKTR_SPIRV_BINARIES "")

foreach(SHADER ${VKTR_SHADER_SOURCES})
  get_filename_component(SHADER_NAME ${SHADER} NAME)
  set(SPIRV_OUTPUT "${VKTR_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

  add_custom_command(
    OUTPUT ${SPIRV_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VKTR_SHADER_OUTPUT_DIR}"
    COMMAND Vulkan::glslc ${SHADER} -o ${SPIRV_OUTPUT}
            --target-env=vulkan1.2
    DEPENDS ${SHADER}
    COMMENT "Compiling VkTR shader: ${SHADER_NAME}"
    VERBATIM
  )
  list(APPEND VKTR_SPIRV_BINARIES ${SPIRV_OUTPUT})
endforeach()

add_custom_target(compile_graphics_shaders ALL DEPENDS ${VKTR_SPIRV_BINARIES})
add_dependencies(${PROJECT_NAME} compile_graphics_shaders)
