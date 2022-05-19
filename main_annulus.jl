using Pkg: Pkg;
Pkg.activate(".");
using RayTracer

# save travel times
function save_matfile(D, gr, nθ, nr, spacing, source; path=pwd())
    degs = Int(360 ÷ nθ)
    # find multiple receiver paths paths 
    receivers_θ = 2.0f0:2.0f0:150.0f0
    receivers_θ = vcat(receivers_θ, reverse(360 .- receivers_θ))
    receivers_r = R
    receivers = [
        closest_point(gr, deg2rad(deg), receivers_r; system=:polar) for deg in receivers_θ
    ]
    paths = [recontruct_path(D.prev, source, r) for r in receivers]

    fname = joinpath(path, "vp_$(degs)degs_$(nr)rad_$(spacing)km_spacing_star2_discont")

    travel_times(D, gr, receivers; isave=true, flname=string(fname, ".csv"))

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
    return close(file)
end

# number of elements in the azimuthal and radial direction
nθ, nr = 720, Int(R ÷ 20)
nθ, nr = 180, 50
spacing = 50

# Instantiate grid
gr, G, halo = init_annulus(nθ, nr; spacing=spacing);

# find source
source_θ = 0.0
source_r = R
source = closest_point(gr, source_θ, source_r; system=:polar)

# Load Vp-Vs Earths Profile
profile = velocity_profile() # AK135
# make velocity interpolant
interpolant = LinearInterpolation(profile.r, profile.Vp)

# Vp and Vs profiles interpolated onto the grid
Vp = interpolate_velocity(gr.r, interpolant)

# Find Shortest path
@time D = bfm(G, halo, source, gr, Vp);
ProfileCanvas.@profview D = bfm(G, halo, source, gr, Vp)

@time D = bfm_gpu(G, halo, source, gr, Vp);

# find multiple receiver paths paths
receivers_θ = 2.0f0:2.0f0:150.0f0
receivers_θ = vcat(receivers_θ, reverse(360 .- receivers_θ))
receivers_r = R
receivers = [
    closest_point(gr, deg2rad(deg), receivers_r; system=:polar) for deg in receivers_θ
]
paths = [recontruct_path(D.prev, source, r) for r in receivers]
paths = [recontruct_path(Array(D.prev), source, r) for r in receivers]

# Plot
plot_paths(gr, paths, source, receivers)

save_matfile(
    D1, gr, nθ, nr, spacing, source; path="/home/albert/Documents/tauP/TauP/RayFiles"
)
save_matfile(D1, gr, nθ, nr, spacing, source)
