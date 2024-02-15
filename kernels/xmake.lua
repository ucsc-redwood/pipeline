add_requires("openmp")

target("kernels")
    set_kind("static")
    add_includedirs("../include")
    add_headerfiles("../include/**/*")
    add_files("./*.cpp")
    add_packages("openmp", "glm")

includes("cuda")
