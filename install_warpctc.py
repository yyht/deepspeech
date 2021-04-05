"""setup.py script for warp-ctc TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
import warnings
from setuptools.command.build_ext import build_ext as orig_build_ext

import tensorflow as tf
enable_gpu = True
lib_ext = ".so"

root_path = os.path.realpath(os.path.dirname(__file__))

print("==root path==", root_path)
 
tf_src_path = "/".join(tf.sysconfig.get_include().split("/")[:-2]+['tensorflow'])

print("==tf_src_path==", tf_src_path)

os.environ['TENSORFLOW_SRC_PATH'] = tf_src_path
os.environ['WARP_CTC_PATH'] = os.path.join(root_path, "tensorflow_binding", "libwarpctc.so")

print("==tf_src_path==", os.path.join(root_path, "tensorflow_binding", "libwarpctc.so"))

cuda_home = ""
for key in os.environ:
  env_path = os.environ[key]
  if 'cuda' in env_path:
    for item in env_path.split(":"):
      if "cuda" in item:
        cuda_home = item
        break

os.environ['CUDA_HOME'] = cuda_home

print("==CUDA_HOME==", cuda_home)

warp_ctc_path = os.environ["WARP_CTC_PATH"]

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
warp_ctc_includes = [os.path.join(root_path, 'include')]
include_dirs = tf_includes + warp_ctc_includes

if tf.__version__ >= '1.4':
  include_dirs += [tf_include + '/../../external/nsync/public']

os.environ['TF_CXX11_ABI'] = "1"
if os.getenv("TF_CXX11_ABI") is not None:
  TF_CXX11_ABI = os.getenv("TF_CXX11_ABI")
else:
  warnings.warn("Assuming tensorflow was compiled without C++11 ABI. "
                  "It is generally true if you are using binary pip package. "
                  "If you compiled tensorflow from source with gcc >= 5 and didn't set "
                  "-D_GLIBCXX_USE_CXX11_ABI=0 during compilation, you need to set "
                  "environment variable TF_CXX11_ABI=1 when compiling this bindings. "
                  "Also be sure to touch some files in src to trigger recompilation. "
                  "Also, you need to set (or unsed) this environment variable if getting "
                  "undefined symbol: _ZN10tensorflow... errors")
  TF_CXX11_ABI = "0"

extra_compile_args = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI]
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if tf.__version__ >= '1.4':
  if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
    extra_link_args = ['-L' + tf.sysconfig.get_lib(), '-ltensorflow_framework']

if (enable_gpu):
  extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
  include_dirs += [os.path.join(os.environ["CUDA_HOME"], 'include')]

  # mimic tensorflow cuda include setup so that their include command work
  if not os.path.exists(os.path.join(root_path, "include")):
    os.mkdir(os.path.join(root_path, "include"))

  cuda_inc_path = os.path.join(root_path, "include/cuda")
  if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != os.environ["CUDA_HOME"]:
    if os.path.exists(cuda_inc_path):
        os.remove(cuda_inc_path)
    os.symlink(os.environ["CUDA_HOME"], cuda_inc_path)
  include_dirs += [os.path.join(root_path, 'include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
  if not os.path.exists(loc):
    print(("Could not find file or directory {}.\n"
             "Check your environment variables and paths?").format(loc),
            file=sys.stderr)
    sys.exit(1)

lib_srcs = ['./deepspeech/tensorflow_binding/src/ctc_op_kernel.cc', 
              './deepspeech/tensorflow_binding/src/warpctc_op.cc']

ext = setuptools.Extension('warpctc_tensorflow.kernels',
                           sources = lib_srcs,
                           language = 'c++',
                           include_dirs = include_dirs,
                           library_dirs = [warp_ctc_path],
                           runtime_library_dirs = [os.path.realpath(warp_ctc_path)],
                           libraries = ['warpctc', 'tensorflow_framework'],
                           extra_compile_args = extra_compile_args,
                           extra_link_args = extra_link_args)

class build_tf_ext(orig_build_ext):
  def build_extensions(self):
    self.compiler.compiler_so.remove('-Wstrict-prototypes')
    orig_build_ext.build_extensions(self)

setuptools.setup(
    name = "warpctc_tensorflow",
    version = "0.1",
    description = "TensorFlow wrapper for warp-ctc",
    url = "https://github.com/baidu-research/warp-ctc",
    author = "Jared Casper",
    author_email = "jared.casper@baidu.com",
    license = "Apache",
    packages = ["warpctc_tensorflow"],
    ext_modules = [ext],
    cmdclass = {'build_ext': build_tf_ext}
)
