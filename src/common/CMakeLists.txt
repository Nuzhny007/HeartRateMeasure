cmake_minimum_required(VERSION 3.5)

project(libCommon)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h
)

add_library(Common ${SOURCE} ${HEADERS})
set_target_properties(Common PROPERTIES FOLDER "libs")
