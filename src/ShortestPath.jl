# module ShortestPath

# include("StructuredGrid.jl")
# export LinearMesh
# export Grid
# export LazyGrid
# export NodalIncidence
# export Point
# export Π
# export grid
# export lazy_grid
# export getindex
# export connectivity
# export nodal_incidence
# export distance

using SparseArrays
# using Polyester
using CUDA
using DelimitedFiles
using Interpolations
using CSV
using DataFrames
using GLMakie

include("GridAnnulus.jl")
include("topology/topology.jl")
include("SSSP/ssspm.jl")
include("SSSP/dijkstra.jl")
include("SSSP/bfm.jl")
include("SSSP/bfm_multiphase.jl")
include("SSSP/bfm_gpu.jl")
include("SSSP/bfm_new.jl")
include("SSSP/rcm.jl")
include("Interpolations/interpolation.jl")
include("plots/plots.jl")

const R = 6371.0

function travel_times(D, gr, receivers; isave=false, flname="")
    travel_time = zeros(length(receivers))
    for (i, receiver) in enumerate(receivers)
        travel_time[i] = D.dist[receiver]
    end

    if isave
        θ = rad2deg.(gr.θ[receivers])
        df = DataFrame(; degree=θ, travel_time=travel_time)
        CSV.write(joinpath(pwd(), flname), df)
    end
end

struct VelProfile{T}
    r::T
    Vp::T
    Vs::T
end

function velocity_profile()
    fl = Float32.(readdlm(joinpath(pwd(), "VelocityProfiles/R_Vp_Vs_AK135.txt")))
    depth = fl[:, 1]
    # convert depth -> radius
    r = maximum(depth) .- depth
    # bundle everything together
    return VelProfile(reverse(r), reverse(fl[:, 2]), reverse(fl[:, 3]))
end

function layers2plot(; npoints=180)
    r = R .- (20.0f0, 35.0f0, 210.0f0, 410.0f0, 660.0f0, 2740.0f0, 2891.5f0, 5100.0f0)
    layers = [circle(npoints, r; pop_end=false) for r in r]
    return layers
end

function interpolate_velocity(
    r, interpolant::T; buffer=1
) where {T<:Interpolations.Extrapolation}
    rlayer = R .- (20.0f0, 35.0f0, 210.0f0, 410.0f0, 660.0f0, 2740.0f0, 2891.5f0)
    V = similar(r)
    buffer_zone = buffer # in km
    for i in eachindex(r)
        if r[i] ∈ rlayer
            # V[i] = 0.5*(interpolant(r[i]+buffer_zone) + interpolant(r[i]-buffer_zone))
            # V[i] = max(interpolant(r[i]+buffer_zone), interpolant(r[i]-buffer_zone))
            V[i] = interpolant(r[i] + buffer_zone)
        else
            V[i] = interpolant(r[i])
        end
    end
    return V
end

struct DualVelocity{T}
    above::T
    below::T
end

function dual_velocity(r, interpolant::T; buffer=1) where {T<:Interpolations.Extrapolation}
    rlayer = R .- (20.0f0, 35.0f0, 210.0f0, 410.0f0, 660.0f0, 2740.0f0, 2891.5f0)
    V = Matrix{Float64}(undef, length(r), 2)
    buffer_zone = buffer # in km
    for i in eachindex(r)
        @inbounds if r[i] ∈ rlayer
            # V[i] = 0.5*(interpolant(r[i]+buffer_zone) + interpolant(r[i]-buffer_zone))
            # V[i] = max(interpolant(r[i]+buffer_zone), interpolant(r[i]-buffer_zone))
            V[i, 1] = interpolant(r[i] - buffer_zone)
            V[i, 2] = interpolant(r[i] + buffer_zone)
        else
            V[i, 1] = V[i, 2] = interpolant(r[i])
        end
    end
    return V
end

# end # module
