# exb_ops 实际上应在 pip install 时编译，这里只是用于简单验证
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()), end=\"\")"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()), end=\"\")"
    OUTPUT_VARIABLE TF_LINK_FLAGS)

message(TF_COMPILE_FLAGS: ${TF_COMPILE_FLAGS})
message(TF_COMPILE_FLAGS: ${TF_LINK_FLAGS})

add_library(exb_ops SHARED exb_ops.cpp)
target_link_libraries(exb_ops cexb_pack)
target_compile_options(exb_ops PRIVATE -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-ignored-qualifiers)
set_target_properties(exb_ops PROPERTIES
    COMPILE_FLAGS ${TF_COMPILE_FLAGS}
    LINK_FLAGS ${TF_LINK_FLAGS})

