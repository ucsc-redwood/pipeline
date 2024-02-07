set_project("cuda_pipeline")

add_rules("mode.debug", "mode.release")

set_languages("cxx17")
set_warnings("all")
set_optimize("fastest")

includes("kernels")
includes("benchmarks")

