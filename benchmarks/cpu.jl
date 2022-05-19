using StaticArrays, SparseArrays, Polyester, CUDA
using CSV, DataFrames

include("/home/albert/Desktop/RayTracer.jl/src/StructuredGrid.jl")
include("/home/albert/Desktop/RayTracer.jl/src/Dijsktra.jl")

function bench(N)
    # θ, ϕ, r (angle from z, angle from x, radius)
    R = 6.371f6
    θ = (deg2rad(70.0f0), deg2rad(110.0f0)) # in radians 0 ≤ θ ≤ π 
    ϕ = (deg2rad(70.0f0), deg2rad(110.0f0)) # in radians 0 ≤ ϕ ≤ 2π 
    c0 = (θ[1], ϕ[1], R - 2.0f6) # origin corner
    c1 = (θ[2], ϕ[2], R) # opposite corner
    nels = (N[1], N[2], N[3]) # number of elements per cartesian axis
    nnods = nels .+ 1 # number of nodes per cartesian axis
    gr = grid(c0, c1, nnods, LinearMesh) # generate grid

    nnods = Π(gr.nnods)
    U = rand(nnods) # Nodal velocity

    G, K = nodal_incidence(gr)

    # G_d, gr_d = move2device(K, U, gr)

    fw = distance3D # custom weight function

    source = 1 #(1, 1, 1) # I = 1
    a = @elapsed dijsktra(G, source, gr, U, fw)
    # b = @elapsed dijsktra_parallel(G, source, gr, U, fw);
    # c = @elapsed BFM(G, source, gr, U, fw);
    b = c = 0.0
    return a, b, c
end

function run_benchs()
    N = [2, 10, 20, 30, 40, 50]
    N = [[2, 2, 2], [250, 250, 50]]
    n = length(N)
    a, b, c = zeros(n), zeros(n), zeros(n)
    for (i, Ni) in enumerate(N)
        a[i], b[i], c[i] = bench(Ni)
    end
    df = DataFrame(; N=N, Dijkstra=a, Radius_stepping=b, BFM=c)
    return CSV.write("CPU_benchmarks_250x250x50.csv", df)
end

run_benchs()
