set_project("cuda_pipeline")

add_rules("mode.debug", "mode.release")

set_languages("cxx17")
set_warnings("all")

add_requires("openmp", "glm", "tbb")

includes("kernels")
includes("benchmarks")

includes("apps")
includes("tests")
