# Fresh environment: never read a developer's root Manifest.toml.
include(joinpath(@__DIR__, "check_release.jl"))
using Pkg, TOML, LinearAlgebra
BLAS.set_num_threads(1)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
root = normpath(joinpath(@__DIR__, ".."))
project = TOML.parsefile(joinpath(root, "Project.toml"))
VERSION >= v"1.11" || error("This alpha requires Julia 1.11 or newer.")
if startswith(get(ENV, "GITHUB_REF", ""), "refs/tags/")
    get(ENV, "GITHUB_REF_NAME", "") == "v" * project["version"] ||
        error("The release tag must match Project.toml's version.")
end
artifact_dir = joinpath(root, ".ci-artifacts")
mkpath(artifact_dir)
mktempdir() do environment
    Pkg.activate(environment)
    try
        Pkg.develop(PackageSpec(path=root))
        Pkg.instantiate()
        Pkg.test(project["name"]; julia_args=["--startup-file=no", "--history-file=no", "--compiled-modules=existing"])
    finally
        for filename in ("Project.toml", "Manifest.toml")
            source = joinpath(environment, filename)
            isfile(source) && cp(source, joinpath(artifact_dir, filename); force=true)
        end
    end
end
