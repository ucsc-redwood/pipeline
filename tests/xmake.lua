for _, file in ipairs(os.files("test_*.cpp")) do
    local name = path.basename(file)
    target(name)
    set_kind("binary")
    add_includedirs("../include")
    add_files(name .. ".cpp")
    add_deps("kernels")
    add_packages("openmp")
    add_tests("pass_output", {
        trim_output = true,
        runargs = "10 9 8 7 6 5 4 3 2 1",
        pass_output = "1 2 3 4 5 6 7 8 9 10"
    })
end
