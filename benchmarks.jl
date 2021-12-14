using MAT
include("src/ShortestPath.jl")

function bench(; nθ = 180, nr = 90, spacing = 10)
    nθ = 90
    nr = 31
    spacing = 1
    # Instantiate grid
    gr = init_annulus(nθ, nr, spacing = spacing)
    constrain2layers!(gr) 
    IM = incidence_matrix(gr)

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
    t = time()
    D1 = bfmtest(IM, source, gr, Vp);
    t = time()-t

    input_memory = 1e-6*(Base.summarysize(IM) + Base.summarysize(gr) + Base.summarysize(Vp))

    println("---------------------------------------------------------")
    println("$(Int(360 ÷ nθ))° × $(Int(R ÷ nr))kms,  spacing = $(spacing); finished in $(t) seconds")
    println("Input parameters for SPM -> $(input_memory) Mb")
    
    save_matfile(D1, gr, nθ, nr, spacing, source, path = joinpath(pwd(), "output"))

end

# save travel times
function save_matfile(D, gr, nθ, nr, spacing, source; path = pwd())
    degs = Int(360 ÷ nθ)
    # find multiple receiver paths paths 
    receivers_θ = 2f0:2f0:150f0
    receivers_θ = vcat(receivers_θ, reverse(360 .-receivers_θ) )
    receivers_r = R
    receivers = [closest_point(gr, deg2rad(deg), receivers_r; system = :polar) for deg in receivers_θ]
    paths = [recontruct_path(D.prev, source, r) for r in receivers]

    fname = joinpath(path, "vp_$(degs)degs_$(nr)rad_$(spacing)km_spacing")
    
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

nθ = [90, 180, 360, 720]
nr = [Int(R ÷ dr) for dr in [200, 100, 50, 20]]
spacing = [1, 5, 10, 20]

nθ = [90, 180]
nr = [Int(R ÷ dr) for dr in [200, 100]]
spacing = [20, 10]

for (i, j) in zip(nθ, nr), k in spacing
    @show i, j, k
    # bench(nθ = i, nr = j, spacing = k)
end
# save_matfile(D1, gr, nθ, nr, spacing, source)