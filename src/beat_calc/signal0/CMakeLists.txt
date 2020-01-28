cmake_minimum_required(VERSION 3.5)

project(libsignal)
set(LIB_SIGNAL_NAME signal0)

set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorColor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/stat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorColor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.h
)

add_library(${LIB_SIGNAL_NAME} SHARED ${SOURCE} ${HEADERS})

set(LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

target_link_libraries(${LIB_SIGNAL_NAME} ${LIBS})

set_target_properties(${LIB_SIGNAL_NAME} PROPERTIES FOLDER "libs")
