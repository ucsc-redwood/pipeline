add_requires("cli11", "spdlog")

target("app")
    set_kind("binary")
    add_includedirs("../include")
    add_files("*.cpp")
    add_packages("cli11", "spdlog", "openmp")
    add_deps("kernels")    
    