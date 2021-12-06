function plot_paths(gr, paths, xs, zs, layers, source, receivers)
    npaths = length(paths)

    f, ax, = lines(xs, zs, color = :black, linewidth=2)
    poly!(ax, Makie.Circle(Point2f0(0, 0), R), color = :lightgreen)
    poly!(ax, Makie.Circle(Point2f0(0, 0), 3479.5f0), color = :orange)
    poly!(ax, Makie.Circle(Point2f0(0, 0), 1270f0), color = :yellow)
    # plot velocity layers
    for l in layers
        lines!(ax, l[1], l[2], color=:black, linewidth=1)
    end

    # plot ray paths
    for i in 1:npaths
        lns = lines!(ax, gr.x[paths[i]], gr.z[paths[i]], color = :red, linewidth=1)
        # plot source and receivers
        scatter!(ax, [gr.x[receivers[i]]], [gr.z[receivers[i]]],  markersize= 10,  color = :black, label = "receiver")
    end
    scatter!(ax, [gr.x[source]], [gr.z[source]],  markersize= 15, color = :black, marker='▴', label = "source")

    # remove grid from plot
    hidedecorations!(ax)
    hidespines!(ax)
    # twitch aspect ratio
    ax.aspect = DataAspect()
    f
end