cmake_minimum_required(VERSION 3.5)

project(libDetectTrack)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SkinDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LKTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EmotionDetection.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/FaceDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SkinDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LKTracker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EmotionDetection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/slog.hpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/infengine_defines.hpp
)

file(GLOB EXT_SRC "extension/*.cpp")
file(GLOB EXT_HDR "extension/*.hpp")
# message(${EXT_SRC})

set(SOURCE ${SOURCE} ${EXT_SRC})
set(HEADERS ${HEADERS} ${EXT_HDR})

if (MSVC)
    set(HEADERS ${HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/w_dirent.h)
endif()

add_library(DetectTrack ${SOURCE} ${HEADERS})
set_target_properties(DetectTrack PROPERTIES FOLDER "libs")
