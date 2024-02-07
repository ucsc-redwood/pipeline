add_requires("benchmark")

target("benchmark")
    set_kind("binary")
    add_includedirs("../include")
    add_files("morton.cpp")
    add_packages("benchmark", "glm", "openmp")
    add_deps("kernels")


