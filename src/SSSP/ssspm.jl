abstract type AbstractSPM end

for algorithm in (:BellmanFordMoore, :Dijkstra, :RadiusStepping)
    @eval begin
        struct $(algorithm){T,M} <: AbstractSPM
            prev::T
            dist::M
        end
    end
end

Base.getindex(spm::AbstractSPM) = spm.prev

function recontruct_path(D, source, receiver)
    prev = D.prev
    path = Int[receiver]
    ipath = prev[receiver]
    while ipath ∉ path
        push!(path, ipath)
        # if !haskey(prev, ipath)
        #     break
        # end
        ipath = prev[ipath]
    end
    push!(path, source)

    return path
end

function recontruct_path(prev::Vector, source, receiver)
    path = Int[receiver]
    ipath = prev[receiver]
    while ipath != source
        push!(path, ipath)
        @inbounds ipath = prev[ipath]
    end
    push!(path, source)

    return path
end
