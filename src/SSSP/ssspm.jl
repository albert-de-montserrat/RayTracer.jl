abstract type AbstractSPM end

struct BellmanFordMoore{T, M} <: AbstractSPM
    prev::T
    dist::M
end

struct Dijkstra{T, M} <: AbstractSPM
    prev::T
    dist::M
end

struct RadiusStepping{T, M} <: AbstractSPM
    prev::T
    dist::M
end

Base.getindex(spm::AbstractSPM) = spm.prev