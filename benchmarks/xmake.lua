add_requires("benchmark")

target("benchmark")
    set_kind("binary")
    add_includedirs("../include")
    add_headerfiles("../include/**/*.hpp")
    add_files("benchmark.cpp")
    add_packages("benchmark", "glm", "openmp")
    add_deps("kernels")

includes("cuda")

-- -- for each file in current directory starts with "bm_" create a benchmark
-- for _, file in ipairs(os.files("bm_*.cpp")) do
--     local name = path.basename(file)
--     target(name)
--         set_kind("binary")
--         add_includedirs("../include")
--         add_headerfiles("../include/**/*.hpp")
--         add_files(file)
--         add_packages("benchmark", "glm", "openmp")
--         add_deps("kernels")
-- end

-- for _, file in ipairs(os.files("bm_*.cu")) do
--     local name = path.basename(file) .. "_gpu"
--     target(name)
--         set_kind("binary")
--         add_includedirs("../include")
--         add_headerfiles("../include/**/*.hpp")
--         add_files(file)
--         add_cugencodes("native")
--         add_packages("benchmark", "glm", "openmp")
--         add_deps("kernels", "gpu_kernels")
-- end
