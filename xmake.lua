set_project("cuda_pipeline")

add_rules("mode.debug", "mode.release")

set_languages("cxx17")
set_warnings("all")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end

add_requires("openmp", "glm")

includes("kernels")
includes("benchmarks")

includes("app")