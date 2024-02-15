add_requires("benchmark")

target("benchmark")
    set_kind("binary")
    add_includedirs("../include")
    add_headerfiles("../include/**/*")
    add_files("benchmark.cpp")
    add_packages("benchmark", "glm", "openmp", "tbb")
    add_deps("kernels")

includes("cuda")
