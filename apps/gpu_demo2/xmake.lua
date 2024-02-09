target("gpu_demo2")
    set_kind("binary")
    add_includedirs("../../include")
    add_headerfiles("../../include/**/*.hpp")
    add_files("*.cu")
    add_packages("cli11", "spdlog", "openmp", "glm")
    add_cugencodes("native")
    add_deps("kernels", "gpu_kernels")    
    -- for host compiler, set openmp
    add_cxxflags("-fopenmp")
    add_cuflags("-Xcompiler -fopenmp")
    