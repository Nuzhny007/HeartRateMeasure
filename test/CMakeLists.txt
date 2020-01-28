cmake_minimum_required(VERSION 3.5)

project(HeartRateMeasure)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/src)

if (EIGEN3_FOUND)
  INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
else()
if (CMAKE_COMPILER_IS_GNUCXX)
  INCLUDE_DIRECTORIES("/usr/include/eigen3")
elseif (MSVC)
  INCLUDE_DIRECTORIES("c:/work/libraries/eigen3")
endif()
endif()

link_directories(${Boost_LIBRARY_DIR})

# ----------------------------------------------------------------------
set(SOURCE
    main.cpp
)

set(HEADERS
)

set(LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${InferenceEngine_LIBRARIES}
    BeatCalc
    Common
    DetectTrack
    EulerianMA
)

add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${LIBS})

if (CMAKE_COMPILER_IS_GNUCXX)
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
elseif(MSVC)
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
