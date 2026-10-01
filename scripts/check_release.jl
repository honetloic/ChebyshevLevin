# Metadata-only preflight. Add --tracked before tagging to catch omitted files.
using TOML
root=normpath(joinpath(@__DIR__,".."))
project=TOML.parsefile(joinpath(root,"Project.toml"))
version=VersionNumber(project["version"])
isempty(version.prerelease) && error("The alpha must have a prerelease version.")
get(project["compat"],"julia","")=="1.11" || error("Expected Julia 1.11 baseline.")
for name in keys(get(project,"deps",Dict()))
    haskey(project["compat"],name) || error("Missing compatibility bound for $name")
end
required=["Project.toml","README.md","LICENSE","CHANGELOG.md",".gitignore",
          ".gitattributes",".github/workflows/ci.yml","scripts/ci.jl","scripts/check_release.jl"]
for folder in ("src","test")
    for (dir,_,files) in walkdir(joinpath(root,folder)), filename in files
        (endswith(filename,".jl") || endswith(filename,".csv")) || continue
        push!(required,replace(relpath(joinpath(dir,filename),root),'\\'=>'/'))
    end
end
if project["name"]=="Teukolsky"
    push!(required,"THIRD_PARTY_NOTICES.md")
elseif project["name"]=="PlungeKerr"
    append!(required,["release/Project.toml","scripts/install_alpha.jl","scripts/alpha_smoke.jl",
        ".github/workflows/install-alpha.yml","examples/data/schwarzschild_radial_reference.csv"])
    release=TOML.parsefile(joinpath(root,"release","Project.toml"))
    for source in values(release["sources"])
        startswith(source["url"],"https://github.com/") || error("Nonportable release source")
        source["rev"]=="v"*project["version"] || error("Release revision/version mismatch")
    end
    for name in ("Teukolsky","ChebyshevLevin")
        project["sources"][name]==release["sources"][name] || error("Dependency revision mismatch")
    end
end
for filename in required
    isfile(joinpath(root,filename)) || error("Missing release file: $filename")
end
tracked=Set(split(readchomp(`git -C $root ls-files`),'\n'))
any(f->startswith(basename(f),"Manifest") && endswith(f,".toml"),tracked) &&
    error("A machine-specific Manifest is still tracked; preserve it locally and untrack it.")
if "--tracked" in ARGS
    missing=sort!(collect(setdiff(Set(required),tracked)))
    isempty(missing) || error("Stage these required files before tagging: "*join(missing,", "))
end
println(project["name"]," ",version,": alpha metadata preflight passed",
        "--tracked" in ARGS ? " (required files tracked)" : " (working tree only)")
