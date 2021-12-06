include("src/ShortestPath.jl")

# define earths boundary (this is just for plotting)
Nsurf = 360 # number of points
xs, zs = circle(Nsurf, R, pop_end = false)

# number of elements in the azimuthal and radial direction
nθ, nr = 180, 30
nθ, nr = 180, 60
# Instantiate grid
gr, G = init_annulus(nθ, nr, spacing = 40, star_levels = 2)
Gsp = graph2sparse(G)

# find source
source_θ = 0.0
source_r = R
source = closest_point(gr, source_θ, source_r; system = :polar)

# Load Vp-Vs Earths Profile
profile = velocity_profile()
# for ploting
layers = layers2plot()
# make velocity interpolant
interpolant = LinearInterpolation(profile.r, profile.Vp)
# Vp and Vs profiles interpolated onto the grid
# Vp = [interpolant_vp(gr.r[i]) for i in 1:gr.nnods]
Vp = interpolate_velocity(round.(gr.r, digits=2), interpolant, buffer = 0)
Vp = dual_velocity(round.(gr.r, digits=2), interpolant, buffer = 1)
# interpolate!(Vp, gr)

# Find Shortest path
@time D1 = bfm(Gsp, source, gr, Vp);

partition = partition_grid(gr)
@time D1 = bfm_multiphase(Gsp, source, gr, Vp, partition, interpolant)

# @time D_vp2 = bfm_gpu(Gsp, source, gr, Vp);
# @time D_vp = Dijkstra(G, source, gr, Vp);

# find multiple receiver paths paths
receivers_θ = 10f0:10f0:150f0
receivers_θ = vcat(receivers_θ, reverse(360 .-receivers_θ) )
receivers_r = R
receivers = [closest_point(gr, deg2rad(deg), receivers_r; system = :polar) for deg in receivers_θ]
paths = [recontruct_path(D1.prev, source, r) for r in receivers]

# Plot
plot_paths(gr, paths, xs, zs, layers, source, receivers)

# save travel times
function save_matfile(D, gr, nθ, nr, source)
    degs = Int(360 ÷ nθ)
    # receivers = [closest_point(gr, deg2rad(deg), R; system = :polar) for deg in 0f0:degs:359f0]
    # paths = [recontruct_path(D.prev, source, r) for r in receivers]

    # find multiple receiver paths paths 
    receivers_θ = 10f0:2f0:150f0
    receivers_θ = vcat(receivers_θ, reverse(360 .-receivers_θ) )
    receivers_r = R
    receivers = [closest_point(gr, deg2rad(deg), receivers_r; system = :polar) for deg in receivers_θ]
    paths = [recontruct_path(D.prev, source, r) for r in receivers]

    fname = "vp_$(degs)degs_$(nr)rad_20km_spacing_NewVersion"
    
    travel_times(D, gr, receivers; isave = true, flname = string(fname, ".csv"))

    file = matopen(string(fname, ".mat"), "w")
    write(file, "x", gr.x)
    write(file, "z", gr.z)
    write(file, "theta", gr.θ)
    write(file, "r", gr.r)
    for (i, p) in enumerate(paths)
        write(file, "x_path$i", gr.x[p])
        write(file, "z_path$i", gr.z[p])
        write(file, "travel_time_path$i", D.dist[p])
    end
    close(file)
end

save_matfile(D1, gr, nθ, nr, source)

# times = [D_vp.dist[r[1]] for r in receivers]

# euclidean_distance = [distance(gr.x[source],gr.z[source], gr.x[p[1]], gr.z[p[1]]) for p in p_vp1]

# error = times .- euclidean_distance./8