# Single-node parallel implementation of the  
# Radius-stepping SSSP algorithm
# implemented with a priority queue and predecessor
# approach (better suited for multi-threading). 
# The graph of nodes V and edges E is represented
# by an adjacency list G
function radius_stepping(Gsp::SparseMatrixCSC, source::Int, gr, U::Vector{T}) where T

    # unpack coordinates
    x, z = gr.x, gr.z

    # number of nodes in the graph
    n = Gsp.n

    # allocate dictionary containing best previous nodes
    p = Vector{Int}(undef, n)

    # Settled queue
    Q = fill(true, n) 
    Q[source] = false
    
    # Frontier queue
    F = fill(false, n) 
    F[source] = true
    
    # initial radius
    Δ = 0.0

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)

    # main loop
    it = 1

    while sum(Q) != 0
        relaxation!(Q, F, Gsp, dist, p, x, z, U)
        Δ = min_distance(Q, dist) # threaded version only worth it for very large problems
        update!(F, Q, dist, Δ)
        it+=1
    end
    
    println("Converged in $it iterations")

    return RadiusStepping(p, dist)
end

function update!(F, Q, dist, Δ)
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] = false
        @inbounds if (Q[i] == true) && (dist[i] ≤ Δ)
            Q[i] = false
            F[i] = true
        end
    end
end

@views function relaxation!(Q, F, G, dist, p, x, z, U)
    Threads.@threads for i in eachindex(F)
        @inbounds if F[i]
            for j in  G.rowval[nzrange(G, i)]
                Q[j] != true && continue
                δ = dist[i] + 2*distance(x[i], z[i], x[j], z[j])/abs(U[i]+U[j])
                if dist[j] > δ
                    dist[j] = δ
                    p[j] = i
                end
            end
        end
    end
end

function min_distance(Q::Vector, dist::Vector)
    d = Inf
    for i in eachindex(Q)
        @inbounds if Q[i]
            di = dist[i]
            if di < d
                d = di
            end
        end
    end
    return d
end

function min_distancet(Q::Vector, dist::Vector{T}) where T
    nt = Threads.nthreads()
    d = fill(Inf, nt)
    n = length(Q)
    blockDim = floor(Int, n/nt)
    left = collect(1:blockDim:n) # should think a way to eliminate these allocations
    right = left .+ (blockDim-1) # should think a way to eliminate these allocations
    right[end] = n

    @sync for ib in 1:nt
        Threads.@spawn for i in left[ib]:right[ib]
            @inbounds if Q[i]
                di = dist[i]
                if dist[i] < d[ib]
                    d[ib] = di
                end
            end
        end
    end
    
    return minimum(d)
end