using MMLDS_project
using Documenter

DocMeta.setdocmeta!(MMLDS_project, :DocTestSetup, :(using MMLDS_project); recursive=true)

makedocs(;
    modules=[MMLDS_project],
    authors=".",
    repo="https://github.com/TobiasGerwald/MMLDS_project.jl/blob/{commit}{path}#{line}",
    sitename="MMLDS_project.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
