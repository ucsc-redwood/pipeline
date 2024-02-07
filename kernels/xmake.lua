add_requires("glm")

add_requires("openmp")

target("kernels")
    set_kind("static")
    add_includedirs("../include") 
    add_headerfiles("../include/**/*.hpp")
    add_files("./morton.cpp")
    add_packages("openmp")