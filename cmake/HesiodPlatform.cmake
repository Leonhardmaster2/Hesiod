add_library(hesiod_platform INTERFACE)

# Linux
if(UNIX AND NOT APPLE)
  message(STATUS "Platform: Linux")
  target_compile_definitions(hesiod_platform INTERFACE HSD_OS_LINUX)

# macOS
elseif(APPLE)
  message(STATUS "Platform: macOS")
  target_compile_definitions(hesiod_platform INTERFACE HSD_OS_MACOS)

# Windows
elseif(WIN32)
  message(STATUS "Platform: Windows")

  # Unsupported platforms
else()
  message(
    FATAL_ERROR
      "Unsupported platform. Only Linux, macOS and Windows are supported.")
endif()
