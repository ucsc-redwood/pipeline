target("benchmark-gpu")
    set_kind("binary")
    add_includedirs("../../include")
    add_headerfiles("../../include/**/*")
    add_files("benchmark.cu")
    add_cugencodes("native")
    add_packages("benchmark", "glm", "openmp")
    add_deps("kernels", "gpu_kernels")

