cmake_minimum_required(VERSION 3.10)

if (NOT OPENEMBEDDING_VERSION)
    set(OPENEMBEDDING_VERSION 0.0.0)
endif()

project(openembedding VERSION ${OPENEMBEDDING_VERSION})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/pico-ps/cmake ${PROJECT_SOURCE_DIR}/pico-ps/pico-core/cmake)

if (NOT PYTHON)
    set(PYTHON "python3")
endif()

if(THIRD_PARTY)
    set(CMAKE_PREFIX_PATH "${THIRD_PARTY}")
    message(STATUS "THIRD_PARTY=${THIRD_PARTY}")
    include_directories(SYSTEM ${THIRD_PARTY}/include)
    link_directories(${THIRD_PARTY}/lib ${THIRD_PARTY}/lib64)
    set(OPENSSL_ROOT_DIR ${THIRD_PARTY}/lib64)
endif()
execute_process(COMMAND ${PYTHON} -c "import sysconfig; print(sysconfig.get_paths()['include'], end='')" OUTPUT_VARIABLE PYTHON_INCLUDE)
include_directories(SYSTEM ${PYTHON_INCLUDE})
message(STATUS "PYTHON_INCLUDE=${PYTHON_INCLUDE}")

# check gcc version
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    message(STATUS "gcc ${GCC_VERSION}")
    if(GCC_VERSION VERSION_GREATER 7 OR GCC_VERSION VERSION_EQUAL 7)
        message(STATUS "C++14 activated.")
    else()
        message(FATAL_ERROR "gcc version should be compatible with tensorflow")
    endif()
else()
    message(FATAL_ERROR "only gcc supported")
endif()

add_definitions(--std=c++14 -Wall -Wextra -Wno-deprecated-declarations -Werror -frecord-gcc-switches -fPIC)
include_directories(${PROJECT_SOURCE_DIR})

option(USE_RDMA "whether build with rdma support" OFF)
if (USE_RDMA)
    add_definitions(-DUSE_RDMA)
    set(RDMA_LIBRARIES rdmacm ibverbs)
    message(STATUS "RDMA enabled")
else()
    message(STATUS "RDMA disabled")
    set(RDMA_LIBRARIES )
endif()

option(USE_DCPMM "whether build with rdma support" OFF)
if (USE_DCPMM)
    add_definitions(-DUSE_DCPMM)
    find_package(PMEM REQUIRED)
    message(STATUS "DCPMM enabled")
else()
    message(STATUS "DCPMM disabled")
endif()


if (DEBUG)
    add_definitions(-O0 -g)
else()
    #add_definitions(-O0 -g)
    #add_definitions(-O3 -DNDEBUG)
    add_definitions(-O3 -g -DNDEBUG) #perf
endif()

set(CMAKE_SHARED_LINKER_FLAGS "-pthread -Wl,--whole-archive -lrt -Wl,--no-whole-archive")
set(CMAKE_EXE_LINKER_FLAGS "-pthread -Wl,--whole-archive -lrt -Wl,--no-whole-archive")

add_definitions(-DOPENEMBEDDING_VERSION="${PROJECT_VERSION}")

find_package(Jemalloc REQUIRED)
find_package(PicoCoreDep REQUIRED)

enable_testing()
add_subdirectory(openembedding)

file(GLOB_RECURSE WHL_SRC LICENSE README.md setup.py MANIFEST.in openembedding/tensorflow/*.py openembedding/*.py)
set(HYPEREMBEDDING_OUT ${CMAKE_CURRENT_BINARY_DIR}/openembedding-${PROJECT_VERSION}.tar.gz)
add_custom_command(
        OUTPUT ${HYPEREMBEDDING_OUT}
        DEPENDS ${WHL_SRC} cexb_pack
        COMMAND rm -rf pypi
        COMMAND mkdir -p pypi
        COMMAND echo __version__ = \\\'${PROJECT_VERSION}\\\' > pypi/openembedding_setup.py
        COMMAND cd ${PROJECT_SOURCE_DIR} && cp -r LICENSE README.md setup.py MANIFEST.in openembedding ${CMAKE_CURRENT_BINARY_DIR}/pypi
        COMMAND cp openembedding/libcexb_pack.so pypi/openembedding
        COMMAND cd pypi && ${PYTHON} setup.py sdist
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(pip_package ALL DEPENDS ${HYPEREMBEDDING_OUT})

add_executable(criteo_preprocess test/criteo_preprocess.cpp)
target_link_libraries(criteo_preprocess pico_core ${PicoCoreDep_LIBRARIES} ${Jemalloc_pic_LIBRARIES})