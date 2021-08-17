#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import setuptools
import setuptools.command.build_ext
import distutils.errors
import distutils.sysconfig
import openembedding_setup


work_path = os.path.dirname(os.path.realpath(__file__)) + '/'
cpp_flags = ['--std=c++14', '-Wall', '-Wextra', '-frecord-gcc-switches', '-fPIC']
link_flags = ['-lcexb_pack', '-L' + work_path + 'openembedding']        
libexb = setuptools.Extension('openembedding.libexb', [])
tensorflow_exb_ops = setuptools.Extension('openembedding.tensorflow.exb_ops', [])


class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        self.build_core_extension()
        self.build_tensorflow_extension()
    
    def build_core_extension(self):
        import pybind11
        libexb.sources = ['openembedding/entry/py_api.cc']
        libexb.extra_compile_args = cpp_flags + ['-I' + pybind11.get_include()]
        libexb.extra_link_args = link_flags
        distutils.sysconfig.customize_compiler(self.compiler)
        self.build_extension(libexb)

    def build_tensorflow_extension(self):
        import tensorflow as tf
        tensorflow_exb_ops.sources = ['openembedding/tensorflow/exb_ops.cpp']
        tensorflow_exb_ops.extra_compile_args = cpp_flags + tf.sysconfig.get_compile_flags()
        tensorflow_exb_ops.extra_link_args = link_flags + tf.sysconfig.get_link_flags()
        distutils.sysconfig.customize_compiler(self.compiler)
        self.build_extension(tensorflow_exb_ops)


import textwrap
setuptools.setup(
    name='openembedding',
    version=openembedding_setup.__version__,
    description='Distributed framework to accelerate training and support serving.',
    author='4paradigm',
    author_email='opensource@4paradigm.com',
    long_description=textwrap.dedent('''\
        OpenEmbedding is a distributed framework to accelerate TensorFlow training and
        support TensorFlow Serving. It uses the parameter server architecture to store
        the Embedding Layer. So that single machine memory is not the limit of model size.
        OpenEmbedding can cooperate with all-reduce framework to support both data parallel
        and model parallel.'''),
    url='https://github.com/4paradigm/OpenEmbedding',
    keywords=['deep learning', 'tensorflow', 'keras', 'AI'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: Apache Software License'],
    python_requires='>=3.6',
    setup_requires=['pybind11'],
    extras_require={'tensorflow':['tensorflow']},
    packages=setuptools.find_packages(),
    package_data={'': [work_path + 'openembedding/libcexb_pack.so']},
    ext_modules=[libexb, tensorflow_exb_ops],
    cmdclass={'build_ext': custom_build_ext})
