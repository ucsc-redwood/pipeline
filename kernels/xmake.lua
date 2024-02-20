add_requires("openmp")

target("kernels")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("./*.cpp")
    add_packages("openmp", "glm")
target_end()

-- if has_package("cuda") then
    includes("cuda")
-- end 
