cmake_minimum_required(VERSION 3.5)

project(libBeatCalc)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/MainProcess.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/MainProcess.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalPlugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.h
)

add_library(BeatCalc ${SOURCE} ${HEADERS})
set_target_properties(BeatCalc PROPERTIES FOLDER "libs")

add_subdirectory(signal0)
add_subdirectory(signal_vpg)
