cmake_minimum_required(VERSION 3.5)

project(libEulerianMA)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/MotionAmp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EulerianMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iir.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/MotionAmp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EulerianMA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleMA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iir.h
)

add_library(EulerianMA ${SOURCE} ${HEADERS})
set_target_properties(EulerianMA PROPERTIES FOLDER "libs")
