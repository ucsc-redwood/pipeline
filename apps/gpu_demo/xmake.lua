target("gpu_demo")
    set_kind("binary")
    add_includedirs("../../include")
    add_headerfiles("../../include/**/*")
    add_files("main.cu")
    add_packages("cli11", "spdlog", "openmp", "glm")
    add_cugencodes("native")
    add_deps("kernels", "gpu_kernels")
    -- for host compiler, set openmp
    add_cxxflags("-fopenmp")
    add_cuflags("-Xcompiler -fopenmp")
