target("gpu_kernels")
    set_kind("static")
    add_includedirs("$(projectdir)/include") 
    add_files("*.cu")
    add_cugencodes("native")
    add_packages("openmp", "tbb", "glm")
