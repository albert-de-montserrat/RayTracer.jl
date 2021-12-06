# Classic Dijkstra Shortest Path (DSP) Algorithm
# implemented with a priority queue. The graph of 
# nodes V and edges E is represented by and adjacency 
# list G. 
# The classic implementation does not leave room for
# parallelization or vectorization.

struct Dijkstra{T, M}
    prev::T
    dist::M
end

const ∞ = Inf

# classic implementation for given precomputed edge weights
function dijkstra(G::Dict, source::Int, ω)

    # number of nodes in the graph
    n = length(G) 

    # allocate dictionary containing best previous nodes
    p = Dict{Int, Int}() 

    # initialise all distances as infinity and zero at the source node 
    dist = fill(Inf, n) 
    dist[source] = 0

    # priority queue containing NOT settled nodes
    Q = Set{Int}() 
    push!(Q, source) # frontier node is the source

    # settled nodes
    settled = Set{Int}() # settled nodes

    # body
    @inbounds while !isempty(Q)
        
        # the frontier node is the one with the mininum distance
        Qi = min_distance(Q, dist) 

        # relax nodes adjacent to the frontier
        for i in G[Qi]

            # skips lines below if i ∈ settled = true
            i ∈ settled && continue 
            
            # temptative distance of successor
            δ = di + ω[i]

            # update distance if it's smaller than 
            # the temptative distance
            if δ < dist[i]
                # update previous node of the path
                p[i] = Qi
                # update distance of successor
                dist[i] = δ
                # push successor to the queuq
                push!(Q, i)
            end
        end

        # remove settled node from the queue
        delete!(Q, Qi)

        # add node to settled set
        push!(settled, Qi)
    end

    return Dijkstra(p, dist)
end

# specialized implementation for on-the-fly weights
function dijkstra(G::Dict, source::Int, gr, U::Vector{T}) where T

    # number of nodes in the graph
    n = length(G) 

    # allocate dictionary containing best previous nodes
    p = Dict{Int, Int}() 

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = 0

    # priority queue containing NOT settled nodes
    Q = Set{Int}() 
    # frontier node is the source
    # note on Set: push!() is faster than union!() 
    # if we need to add a single value to the Set
    push!(Q, source)

    # settled nodes
    settled = Set{Int}() 

    # body
    it = 1
    @inbounds while !isempty(Q)
        
        # the frontier node is the one with the mininum distance
        Qi = min_distance(Q, dist) 

        # # cache coordinates, velocity and distance of frontier node
        # xi, zi, Ui, di = gr.x[Qi], gr.z[Qi], U[Qi], dist[Qi]

        # # relax nodes adjacent to the frontier
        # for i in G[Qi]

        #     # skips lines below if i ∈ settled = true
        #     i ∈ settled && continue 
            
        #     # temptative distance of successor
        #     δ = di + 2*distance(xi, zi, gr.x[i], gr.z[i])/abs(U[i]+Ui)

        #     # update distance if it's smaller than 
        #     # the temptative distance
        #     if δ < dist[i]
        #         # update previous node of the path
        #         p[i] = Qi
        #         # update distance of successor
        #         dist[i] = δ
        #         # push successor to the queuq
        #         push!(Q, i)
        #     end
        # end

        _relax_dijkstra!(p, dist, Q, gr, G, Qi, U, settled)

        # remove settled node from the queue
        delete!(Q, Qi)

        # add node to settled set
        push!(settled, Qi)

        # update iteration counter
        it += 1
    end

    println("Converged in $it iterations")

    return Dijkstra(p, dist)
end

@views function _relax_dijkstra!(p, dist, Q, gr, G, Qi, U, settled)
    # cache coordinates, velocity and distance of frontier node
    xi, zi, Ui, di = gr.x[Qi], gr.z[Qi], U[Qi], dist[Qi]

    # relax nodes adjacent to the frontier
    for i in G[Qi]

        # skips lines below if i ∈ settled = true
        i ∈ settled && continue 
            
        # temptative distance of successor
        δ = di + 2*distance(xi, zi, gr.x[i], gr.z[i])/abs(U[i]+Ui)

        # update distance if it's smaller than 
        # the temptative distance
        if δ < dist[i]
            # update previous node of the path
            p[i] = Qi
            # update distance of successor
            dist[i] = δ
            # push successor to the queuq
            push!(Q, i)
        end

    end
end

function min_distance(Q::Set{Int}, dist::Vector{T}) where T
    d = typemax(T) # ∞ of appropriate type
    idx = -1

    # iterate over queue to find index corresponding
    # to the minimum distance
    for Qi in Q
        @inbounds di = dist[Qi]
        if di < d
            d = di
            idx = Qi
        end
    end
    return idx
end

function recontruct_path(D, source, receiver)
    prev = D.prev
    path = Int[receiver]
    ipath = prev[receiver]
    while ipath ∉ path
        # while ipath != source
        push!(path, ipath)
        # if !haskey(prev, ipath)
        #     break
        # end
        @show ipath = prev[ipath]
    end
    push!(path, source)

    return path
end

@inbounds function recontruct_path(prev::Vector, source, receiver)
    path = Int[receiver]
    ipath = prev[receiver]
    while ipath != source
        push!(path, ipath)
        ipath = prev[ipath]
        # @show ipath
    end
    push!(path, source)

    return path
end