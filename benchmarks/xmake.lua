add_requires("benchmark")

-- for each file in current directory starts with "bm_" create a benchmark
for _, file in ipairs(os.files("bm_*.cpp")) do
    local name = path.basename(file)
    target(name)
        set_kind("binary")
        add_includedirs("../include")
        add_headerfiles("../include/**/*.hpp")
        add_files(file)
        add_packages("benchmark", "glm", "openmp")
        add_deps("kernels")
end
