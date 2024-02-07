add_requires("glm")

add_requires("openmp", "tbb")

target("kernels")
    set_kind("static")
    add_includedirs("../include") 
    add_headerfiles("../include/**/*.hpp")
    add_files("./*.cpp")
    add_packages("openmp", "tbb")