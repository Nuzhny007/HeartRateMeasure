cmake_minimum_required(VERSION 2.8)

project(VitaGraph)

cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0028 NEW)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find the QtWidgets library
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Xml REQUIRED)

include_directories(${QT_INCLUDE_DIRS} 
                    ${CMAKE_SOURCE_DIR}/src)


link_directories(${Boost_LIBRARY_DIR})

aux_source_directory(src SRC_LIST)

qt5_add_resources(RESOURCE_ADDED VitaGraph.qrc)

set(SOURCE
    main.cpp
    MainWindow.cpp
    UtilityFunctions.cpp
    VideoFrame.cpp
    VideoScene.cpp
)

set(HEADERS
    MainWindow.h
    UtilityFunctions.h
    VideoFrame.h
    VideoScene.h
)

set(LIBS
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets 
    Qt5::Xml
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${InferenceEngine_LIBRARIES}
    BeatCalc
    Common
    DetectTrack
    EulerianMA
)

if (CMAKE_COMPILER_IS_GNUCXX)
add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS} ${RESOURCE_ADDED})
elseif (MSVC)
add_executable(${PROJECT_NAME} WIN32 ${SOURCE} ${HEADERS} ${RESOURCE_ADDED} VitaGraph.rc)
endif()

target_link_libraries(${PROJECT_NAME} ${LIBS})

if (CMAKE_COMPILER_IS_GNUCXX)
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
elseif(MSVC)
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
