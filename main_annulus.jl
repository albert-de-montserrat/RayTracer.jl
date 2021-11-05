using StaticArrays
using SparseArrays
using Polyester
using CUDA
using DelimitedFiles
using Interpolations
using CSV
using DataFrames
# using GLMakie

include("src/GridAnnulus.jl")
include("src/SSSP/dijkstra.jl")
include("src/SSSP/bfm.jl")
include("src/SSSP/bfm_gpu.jl")
 
function travel_times(D, gr, receivers; isave = false, flname = "")
    travel_time = zeros(length(receivers))
    for (i, receiver) in enumerate(receivers)
        travel_time[i] = D.dist[receiver]
    end

    if isave
        # writedlm(flname, hcat( rad2deg.(gr.θ[receivers.+1]), travel_time))
        θ = rad2deg.(gr.θ[receivers])
        df = DataFrame(degree=θ, travel_time = travel_time)
        CSV.write(flname,df)
    end

end

struct VelProfile{T}
    r::T
    Vp::T
    Vs::T
end

function velocity_profile()
    fl = Float32.(readdlm("VelocityProfiles/R_Vp_Vs_AK135.txt"))
    depth = fl[:,1]
    # convert depth -> radius
    r = maximum(depth) .- depth
    # bundle everything together
    return VelProfile(reverse(r), reverse(fl[:,2]), reverse(fl[:,3]))
end

R = 6371f0
# Instantiate grid
N = 100
Nsurf = 180
nθ, nr = Nsurf, Int(R÷N)
nθ, nr = 90, 20
# define earths boundary
xs, zs = circle(Nsurf, R, pop_end = false)
gr, G, Gsp = init_annulus(nθ, nr)

# gr.nnods

# find source
source = closest_point(gr, 0f0, R; system = :polar)

# Load Vp-Vs Earths Profile
profile = velocity_profile()
# make velocity interpolant
interpolant_vp = LinearInterpolation(profile.r, profile.Vp)
interpolant_vs = LinearInterpolation(profile.r, profile.Vs)
# Vp and Vs profiles interpolated onto the grid
Vp = [interpolant_vp(gr.r[i]) for i in 1:gr.nnods]
Vs = [interpolant_vs(gr.r[i]) for i in 1:gr.nnods] 

# Find Shortest path
receiver = closest_point(gr, deg2rad(90f0), R; system = :polar)
@time D_vp = dijkstra(G, source, gr, Vp);
@time D_vp = bfm(Gsp, source, gr, Vp);
@time D_vp = bfm_gpu(Gsp, source, gr, Vp);

# save travel times
receivers = [closest_point(gr, deg2rad(deg), R; system = :polar) for deg in 0f0:360f0]
degs = Int(360 ÷ nθ)
travel_times(D_vp, gr, receivers; isave = false, flname = "vp_$(degs)degs_$(nr)rad.csv")

# Reconstruct ray paths
p_vp = recontruct_path(D_vp, source, receiver)
# multiple paths 
receivers = [closest_point(gr, deg2rad(deg), R; system = :polar) for deg in 10f0:5f0:150f0]
p_vp1 = [recontruct_path(D_vp.prev, source, r) for r in receivers]

# receivers = [closest_point(gr, deg2rad(deg+180), R; system = :polar) for deg in 10f0:5f0:150f0]
# p_vp2 = [recontruct_path(D_vp, source, r) for r in receivers]

# Coordinates of the ray path
vp_x, vp_z = gr.x[p_vp1], gr.z[p_vp1]

# Plot
f, ax, = lines(xs, zs, color = :black, markersize = 2)
scatter!(gr.x,gr.z)
# plot ray paths
lines!(ax, vp_x, vp_z, color=:red, linewidth=3)
# lines!(ax, vs_x, vs_z, color=:green, linewidth=3)
# plot source and receivers
scatter!(ax, [gr.x[source]], [gr.z[source]],  markersize= 15, color = :magenta, marker='■', label = "source")
scatter!(ax, [gr.x[receiver]], [gr.z[receiver]],  markersize= 15, color = :magenta, marker='▴', label = "receiver")
# remove grid from plot
hidedecorations!(ax, ticklabels = false, ticks = false)
# twitch aspect ratio
ax.aspect = DataAspect()
f

# Plot
npaths=length(p_vp1)
x2900, z2900 = circle(N, 6371.0f0 - 2900f0, pop_end = false) 

f, ax, = lines(xs, zs, color = :black, linewidth=3)
# plot ray paths
for i in 1:npaths
    lines!(ax, gr.x[p_vp1[i]], gr.z[p_vp1[i]], color=:blue, linewidth=1)
    # lines!(ax, gr.x[p_vp2[i]], gr.z[p_vp2[i]], linewidth=3)
    # plot source and receivers
    scatter!(ax, [gr.x[receivers[i]]], [gr.z[receivers[i]]],  markersize= 15, color = :magenta, marker='▴', label = "receiver")
    # scatter!(ax, [-gr.x[receivers[i]]], [gr.z[receivers[i]]],  markersize= 15, color = :magenta, marker='▴', label = "receiver")
end
scatter!(ax, [gr.x[source]], [gr.z[source]],  markersize= 15, color = :magenta, marker='■', label = "source")
lines!(ax, x2900, z2900, color=:orange, linewidth=3)

# remove grid from plot
# hidedecorations!(ax, ticklabels = false, ticks = false)
# hidedecorations!(ax)
# hidespines!(ax)
# xlabel!(ax, )
# twitch aspect ratio
ax.aspect = DataAspect()
ax.xticks = ([-R,0,R], ["-R","0","R"])
ax.yticks = ([-R,0,R], ["-R","0","R"])
f
