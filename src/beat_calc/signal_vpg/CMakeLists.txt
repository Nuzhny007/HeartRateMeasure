cmake_minimum_required(VERSION 3.5)

project(libsignal_vpg)
set(LIB_SIGNAL_NAME signal_vpg)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/VPGSignalProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hrvprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/peakdetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pulseprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/VPGSignalProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/hrvprocessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/peakdetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pulseprocessor.h
)

add_library(${LIB_SIGNAL_NAME} SHARED ${SOURCE} ${HEADERS})

set(LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

target_link_libraries(${LIB_SIGNAL_NAME} ${LIBS})

set_target_properties(${LIB_SIGNAL_NAME} PROPERTIES FOLDER "libs")
