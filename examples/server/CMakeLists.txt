set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET qwen-asr-server)
add_executable(${TARGET} server.cpp httplib.h)

# Include paths: qwen-asr C headers + current dir for httplib.h and json.hpp
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against the qwen_asr static library / object files and system libs
target_link_libraries(${TARGET} PRIVATE
    qwen_asr
    ${CMAKE_THREAD_LIBS_INIT}
)

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

install(TARGETS ${TARGET} RUNTIME)
