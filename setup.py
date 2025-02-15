import os
import re
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is installed.
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_fullpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        extdir = os.path.dirname(ext_fullpath)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}"
        ]
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_temp)

setup(
    name="gdcpp",
    version="0.1.0",
    author="Giacomo Amerio",
    description="Python bindings for Gradient Descent Optimizer",
    ext_modules=[CMakeExtension("gdcpp", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)