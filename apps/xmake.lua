add_requires("cli11", "spdlog")

includes("cpu_demo")

-- if has_package("cuda") then
    includes("gpu_demo")
    includes("ppl")
-- end
