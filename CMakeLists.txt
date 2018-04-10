cmake_minimum_required (VERSION 2.8)

project (HeartRateMeasure)

FIND_PACKAGE(OpenCV 3.4 REQUIRED)

FIND_PACKAGE(Eigen3)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

unset(CMAKE_CXX_FLAGS CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors --std=c++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4" CACHE STRING COMPILE_FLAGS FORCE)
endif()

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

if (EIGEN3_FOUND)
  INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
  message(${EIGEN3_INCLUDE_DIR})
else()
  INCLUDE_DIRECTORIES("/usr/include/eigen3")
  INCLUDE_DIRECTORIES("c:/alles/libraries/eigen3")

#  get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#  foreach(dir ${dirs})
#    message(STATUS "dir='${dir}'")
#  endforeach()
endif()

link_directories(${Boost_LIBRARY_DIRS})

# ----------------------------------------------------------------------
set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorColor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorMoving.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/FaceDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/SkinDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/LKTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/EulerianMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/iir.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/stat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorColor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessorMoving.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/FaceDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/SkinDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/LKTracker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/EulerianMA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/iir.h
)

set(LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

add_executable(HeartRateMeasure ${SOURCE} ${HEADERS})

target_link_libraries(HeartRateMeasure ${LIBS})
