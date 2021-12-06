include("src/ShortestPath.jl")
using MAT

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
profile = velocity_profile() # AK135
# make velocity interpolant
interpolant = LinearInterpolation(profile.r, profile.Vp)

# Vp and Vs profiles interpolated onto the grid
Vp = interpolate_velocity(round.(gr.r, digits=2), interpolant, buffer = 0)
Vp = dual_velocity(round.(gr.r, digits=2), interpolant, buffer = 1)

# Find Shortest path
@time D1 = bfm(Gsp, source, gr, Vp);
# @time D_vp2 = bfm_gpu(Gsp, source, gr, Vp);
# @time D_vp = Dijkstra(G, source, gr, Vp);

# find multiple receiver paths paths
receivers_θ = 10f0:10f0:150f0
receivers_θ = vcat(receivers_θ, reverse(360 .-receivers_θ) )
receivers_r = R
receivers = [closest_point(gr, deg2rad(deg), receivers_r; system = :polar) for deg in receivers_θ]
paths = [recontruct_path(D1.prev, source, r) for r in receivers]

# Plot
# define earths boundary
Nsurf = 360 # number of points
xs, zs = circle(Nsurf, R, pop_end = false)
layers = layers2plot()
plot_paths(gr, paths, xs, zs, layers, source, receivers)

# save travel times
function save_matfile(D, gr, nθ, nr, source)
    degs = Int(360 ÷ nθ)
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