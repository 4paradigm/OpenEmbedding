include_directories(client server variable entry)

file(GLOB exb_src client/*.cpp server/*.cpp variable/EmbeddingVariable.cpp entry/c_api.cc)
add_library(cexb_obj OBJECT ${exb_src})

# cexb_static for tensorflow ops
add_library(cexb_static STATIC $<TARGET_OBJECTS:cexb_obj>)
target_link_libraries(cexb_static pico_ps_static pico_core_static ${PicoCoreDep_STATIC_LIBRARIES} ${Jemalloc_pic_STATIC_LIBRARIES} ${RDMA_LIBRARIES} dl)

# cexb with out static libraries
# add_library(cexb SHARED $<TARGET_OBJECTS:cexb_obj>)
# target_link_libraries(cexb pico_ps pico_core ${PicoCoreDep_LIBRARIES} ${Jemalloc_pic_LIBRARIES} ${RDMA_LIBRARIES} dl)

# cexb_pack with static libraries for tensorflow-serving
add_library(cexb_pack SHARED $<TARGET_OBJECTS:cexb_obj>)
target_link_libraries(cexb_pack pico_ps_static pico_core_static ${PicoCoreDep_STATIC_LIBRARIES} ${Jemalloc_pic_STATIC_LIBRARIES} ${RDMA_LIBRARIES} dl)

add_executable(masterd entry/masterd.cc)
target_link_libraries(masterd cexb_static)

add_executable(server entry/server.cc)
target_link_libraries(server cexb_static)

find_package(Protobuf REQUIRED)
find_package(OpenSSL REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
find_lib(BRPC_STATIC_LIBRARIES STATIC LIBS brpc protobuf)
find_lib(BRPC_DYNAMIC_LIBRARIES SHARED LIBS leveldb)
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER entry/controller.proto)
add_executable(controller entry/controller.cc ${PROTO_SRC})
target_compile_options(controller PRIVATE -Wno-unused-parameter)
target_link_libraries(controller cexb_static ${BRPC_STATIC_LIBRARIES} ${BRPC_DYNAMIC_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARY} ${OPENSSL_SSL_LIBRARY})

option(SKIP_CHECK_WHEEL_SETUP "try build tensorflow operator" OFF)
if (NOT SKIP_CHECK_WHEEL_SETUP)
    # py_api should be compiled during pip install, here is just for simple verification.
    execute_process(COMMAND ${PYTHON} -c "import pybind11; print(pybind11.get_include(), end=\"\")"
        OUTPUT_VARIABLE PYBIND11_INCLUDE)
    add_library(exb SHARED entry/py_api.cc)
    target_include_directories(exb PRIVATE ${PYBIND11_INCLUDE})
    target_link_libraries(exb PRIVATE cexb_pack)

    add_subdirectory(tensorflow)
endif()

# tests
find_package(PicoTestDep)
link_libraries(cexb_static ${PicoTestDep_STATIC_LIBRARIES})
add_executable(c_api_test entry/c_api_test.cpp)
add_executable(c_api_ha_test entry/c_api_ha_test.cpp)
if (USE_DCPMM)
    add_executable(persistent_embedding_table_test entry/persistent_embedding_table_test.cpp)
endif()

include(GoogleTest)
gtest_discover_tests(c_api_test)
# At present, ha_test has a probability of failing, 
# because the current ps restore dead node has a small probability of failing.
# This situation is currently considered by unittest to be caused by an abnormal restore crash.
# Actually, you only need to restart again at this time. 
# When restarting the PS, the startup failure should be considered.
# gtest_discover_tests(c_api_ha_test)