using CSV, DataFrames
path = joinpath(pwd(), "benchmarks/")
fls = ["GPU_1_threads.csv", "GPU_4_threads.csv", "GPU_6_threads.csv"]

fls = ["GPU_1_threads_star1_NOreorder_new.csv", "GPU_4_threads_star1_NOreorder_new.csv"]

fls = joinpath.(path, fls)

df1 = CSV.read(fls[1], DataFrame)
df2 = CSV.read(fls[2], DataFrame)
# df3 = CSV.read(fls[3], DataFrame)

dijsktra = df1.Dijkstra[2:end]
bfm_1thread = df1.bfm_cpu[2:end]
bfm_1gpu = df2.bfm_gpu[2:end]
bfm_4thread = df2.bfm_cpu[2:end]
# bfm_6thread = df2.bfm_cpu[2:end]
nn = (df1.nodes[2:end])

df = DataFrame(;
    nodes=nn,
    dijsktra=dijsktra,
    bfm_1thread=bfm_1thread,
    bfm_4thread=bfm_4thread,
    # bfm_6thread = bfm_6thread,
    bfm_1gpu=bfm_1gpu,
)

CSV.write("bench_data2.csv", df)

f = Figure()

ax = Axis(
    f[1, 1];
    xlabel="log10(nodes)",
    ylabel="time [s]",
    yscale=log10,
    yminorticksvisible=true,
    yminorgridvisible=true,
    yminorticks=IntervalsBetween(8),
)
lines!(ax, nn, dijsktra; label="Dijkstra", linewidth=2)
lines!(ax, nn, bfm_1thread; label="BFM sequential", linewidth=2)
lines!(ax, nn, bfm_4thread; label="BFM 4 threads", linewidth=2)
lines!(ax, nn, bfm_6thread; label="BFM 6 threads", linewidth=2)
lines!(ax, nn, bfm_1gpu; label="BFM GeForce GTX 1660 Ti", linewidth=2)
axislegend(ax; position=:lt)
hidexdecorations!(ax; grid=false)

ax = Axis(f[2, 1]; xlabel="log10(nodes)", ylabel="speed up")
lines!(ax, nn, dijsktra ./ bfm_1gpu; label="BFM GPU vs Dijkstra", linewidth=2)
lines!(ax, nn, bfm_1thread ./ bfm_1gpu; label="BFM GPU vs CPU", linewidth=2)
axislegend(ax; position=:rb)
f
