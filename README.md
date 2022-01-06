# RayTracer.jl
## Current dependencies 

1. SparseArrays
2. Interpolations
3. CUDA ~(experimental)~
4. DelimitedFiles
5. to generate output: CSV, DataFrames, MAT (MATLAB output format shall be deprecated in release version)

## TODO list
- [x] Optimize sparse graph (moved from SparseArray to custom data structured + parallelized)
- [x] ~Optimize generation of adjacency list~ Rework memory layout of adjency list
- [ ] Re-write Cuthill-McKee algorithm (node re-ordering)
- [ ] 3D mesh wrapper

## Example 2D annulus
```julia
include("src/ShortestPath.jl")

# Instantiate grid
nθ, nr = 180, 50
spacing = 1 # [km] distance between secondary nodes
gr, G, halo = init_annulus(nθ, nr, spacing = spacing);

# find source
source_θ, = 0.0, R # sorce node coordinate
source = closest_point(gr, source_θ, source_r; system = :polar)

# Load Vp-Vs Earths Profile
profile = velocity_profile() # AK135 by default
# make velocity interpolant
interpolant = LinearInterpolation(profile.r, profile.Vp)

# Vp and Vs profiles interpolated onto the grid
Vp = interpolate_velocity(gr.r, interpolant)

# Find Shortest path
D = bfm(G, halo, source, gr, Vp); # Cpu version ~OpenMP 
Dgpu = bfm_gpu(G, halo, source, gr, Vp); # GPU version

# find multiple receiver paths paths
receivers_θ = 10f0:10f0:150f0
receivers_θ = vcat(receivers_θ, reverse(360 .-receivers_θ) )
receivers_r = R
receivers = [closest_point(gr, deg2rad(deg), receivers_r; system = :polar) for deg in receivers_θ]
paths = [recontruct_path(D.prev, source, r) for r in receivers]

# Plot
# define earths boundary
Nsurf = 360 # number of points
xs, zs = circle(Nsurf, R, pop_end = false)
layers = layers2plot()
plot_paths(gr, paths, xs, zs, layers, source, receivers)
```

![output](ray_paths.png)
