add_requires("openmp")

target("ppl")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("*.hpp", "**/*.cuh", "../../include/**/*")
    add_files("main.cu")
    add_cugencodes("native")
    add_packages("glm", "spdlog", "cli11", "openmp")
    add_deps("kernels", "gpu_kernels")
    -- for host compiler, set openmp
    add_cxxflags("-fopenmp")
    add_cuflags("-Xcompiler -fopenmp")
