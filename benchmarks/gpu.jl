using StaticArrays, SparseArrays, Polyester, CUDA, CUDA.CUSPARSE
using CSV, DataFrames, DelimitedFiles, Interpolations

include("/home/albert/Desktop/RayTracer.jl/src/GridAnnulus.jl")
include("/home/albert/Desktop/RayTracer.jl/src/SSSP/dijkstra.jl")
include("/home/albert/Desktop/RayTracer.jl/src/SSSP/bfm.jl")
include("/home/albert/Desktop/RayTracer.jl/src/SSSP/bfm_gpu.jl")
include("/home/albert/Desktop/RayTracer.jl/src/SSSP/rcm.jl")
include("/home/albert/Desktop/RayTracer.jl/src/Interpolations/interpolation.jl")
include("/home/albert/Desktop/RayTracer.jl/src/Interpolations/barycentric.jl")
include("/home/albert/Desktop/RayTracer.jl/src/Interpolations/bilinear.jl")

const R = 6371.0f0

function travel_times(D, gr, receivers; isave=false, flname="")
    travel_time = zeros(length(receivers))
    for (i, receiver) in enumerate(receivers)
        travel_time[i] = D.dist[receiver]
    end

    if isave
        θ = rad2deg.(gr.θ[receivers])
        df = DataFrame(; degree=θ, travel_time=travel_time)
        CSV.write(flname, df)
    end
end

struct VelProfile{T}
    r::T
    Vp::T
    Vs::T
end

function velocity_profile()
    fl = "/home/albert/Desktop/ShortestPath/VelocityProfiles/R_Vp_Vs_AK135.txt"
    fl = Float32.(readdlm(fl))
    depth = fl[:, 1]
    # convert depth -> radius
    r = maximum(depth) .- depth
    # bundle everything together
    return VelProfile(reverse(r), reverse(fl[:, 2]), reverse(fl[:, 3]))
end

function bench(nθ, nr)

    # define earths boundary
    gr, G = init_annulus(nθ, nr)
    Gsp = graph2sparse(G)

    # find source
    source = closest_point(gr, 0.0f0, R; system=:polar)

    # Load Vp-Vs Earths Profile
    profile = velocity_profile()
    # make velocity interpolant
    interpolant_vp = LinearInterpolation(profile.r, profile.Vp)
    Vp = [interpolant_vp(gr.r[i]) for i in 1:(gr.nnods)]
    interpolate!(Vp, gr)

    # Find Shortest path
    a = @elapsed dijkstra(G, source, gr, Vp)
    b = @elapsed bfm(Gsp, source, gr, Vp)
    # c = @elapsed bfm_gpu(Gsp, source, gr, Vp);

    return length(gr.x), a, b
end

function run_benchs()
    nθ = ([5, 90, 180, 180])
    nr = ([5, 32, 63, 90])
    n = length(nr)
    a, b, c = zeros(n), zeros(n), zeros(n)
    nnodes = zeros(n)
    for i in 1:n
        nnodes[i], a[i], b[i], = bench(nθ[i], nr[i])
        # b[i] = bench(nθ[i], nr[i])
    end

    df = DataFrame(;
        nodes=nnodes,
        nθ=nθ,
        nr=nr,
        Dijkstra=a,
        bfm_cpu=b,
        # bfm_gpu = c,
    )

    nt = Threads.nthreads()
    return CSV.write("GPU_$(nt)_threads_star1_NOreorder_new.csv", df)
end

run_benchs()
