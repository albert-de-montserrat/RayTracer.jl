# Single-node parallel implementation of the  
# Bellman-Ford-Moore (BFM) SSSP algorithm
# implemented with a priority queue and sucessor
# approach (better suited for multi-threading). 
# The graph of nodes V and edges E is represented
# by an adjacency list G

struct BellmanFordMoore{T, M}
    prev::T
    dist::M
end

# specialized implementation for on-the-fly weights
function bfm(G::Dict{M, Set{M}}, source::Int, gr, U::Vector{T}) where {M,T}

    # unpack coordinates
    x, z = gr.x, gr.z

    # number of nodes in the graph
    n = length(G) 

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
    p = Vector{M}(undef, n)

    # # priority queue containing NOT settled nodes
    # Q = Set{M}() 
    # # 1st frontier nodes are nodes adjacent to the source
    # union!(Q, G[source])

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    for Gi in G[source]
        Q[Gi] = true
    end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main lopp
    it = 1
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        _relax_bfm!(dist, p, dist0, G, Q, x, z, U)
            
        # pop queue (serial-but-fast process)
        # empty!(Q)
        fillfalse!(Q)

        # update nodal queue (parallel process)
        _update_bfm!(Q, G, dist, dist0)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

function bfm(Gsp::SparseMatrixCSC, source::Int, gr, U::Vector{T}) where T

    # unpack coordinates
    x, z = gr.x, gr.z

    # number of nodes in the graph
    n = Gsp.n

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
    p = Vector{Int}(undef, n)

    # # priority queue containing NOT settled nodes
    # Q = Set{M}() 
    # # 1st frontier nodes are nodes adjacent to the source
    # union!(Q, G[source])

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    for Gi in Gsp.rowval[nzrange(Gsp, source)]
        Q[Gi] = true
    end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main lopp
    it = 1
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        _relax_bfm!(dist, p, dist0, Gsp, Q, x, z, U)
            
        # pop queue (serial-but-fast process)
        # empty!(Q)
        fillfalse!(Q)

        # update nodal queue (parallel process)
        _update_bfm!(Q, Gsp, dist, dist0)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

fillfalse!(A) = fill!(A, false)

@inbounds function _relax_bfm!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, x, z, U) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        
        if Q[i]
            # cache coordinates, velocity and distance of frontier node
            # @inbounds xi, zi, Ui, di = x[i], z[i], U[i], dist0[i]
            @inbounds di = dist0[i]

            # iterate over adjacent nodes to find the the one with 
            # the mininum distance to current node
            for Gi in @views G.rowval[nzrange(G, i)]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                   dist0[Gi] == typemax(T),
                   typemax(T),
                   dist0[Gi] + 2*distance(x[i], z[i], x[Gi], z[Gi])/abs(U[i]+U[Gi])
                )

                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if di > δ
                    di = δ
                    @inbounds p[i] = Gi
                end
            end

            # update distance
            @inbounds dist[i] = di 
        end
    end
end

@inbounds function _relax_bfm!(dist::Vector{T}, p::Vector, dist0, G, Q::BitVector, x, z, U) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        
        if Q[i]
            # cache coordinates, velocity and distance of frontier node
            # @inbounds xi, zi, Ui, di = x[i], z[i], U[i], dist0[i]
            @inbounds di = dist0[i]

            # iterate over adjacent nodes to find the the one with 
            # the mininum distance to current node
            for Gi in G[i]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                   dist0[Gi] == typemax(T),
                   typemax(T),
                   dist0[Gi] + 2*distance(x[i], z[i], x[Gi], z[Gi])/abs(U[i]+U[Gi])
                )

                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if di > δ
                    di = δ
                    @inbounds p[i] = Gi
                end
            end

            # update distance
            @inbounds dist[i] = di 
        end
    end
end

function _relax_bfm!(dist::Vector{T}, p::Vector, dist0, G, Q::Set, x, z, U) where T
    # iterate over quere. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in collect(Q)
        
        # cache coordinates, velocity and distance of frontier node
        @inbounds xi, zi, Ui, di = x[i], z[i], U[i], dist0[i]

        # iterate over adjacent nodes to find the the one with 
        # the mininum distance to current node
        for Gi in G[i]
            # temptative distaance (ignore if it's ∞)
            δ = ifelse(
                dist0[Gi] == typemax(T),
                typemax(T),
                dist0[Gi] + 2*distance(xi, zi, x[Gi], z[Gi])/abs(U[i]+Ui)
            )

            # update distance and predecessor index 
            # if it's smaller than the temptative distance
            if di > δ
                di = δ
                @inbounds p[i] = Gi
            end
        end

        # update distance
        @inbounds dist[i] = di 
    end
end

function _update_bfm!(Q::Set, G, dist, dist0)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    # lk = ReentrantLock()
    # Threads.@threads 
    for i in 1:length(G)
        @inbounds if dist[i] < dist0[i]
            # need to use lock to define atomic
            # region to avoid race condition
            # lock(lk) 
            union!(Q, G[i])
            # unlock(lk)
        end
    end
end

function _update_bfm!(Q::BitVector, G, dist, dist0)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        @inbounds if dist[i] < dist0[i]
            for Gi in G[i]
                Q[Gi] = true
            end
        end
    end
end

function _update_bfm!(Q::BitVector, G::SparseMatrixCSC, dist, dist0)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        @inbounds if dist[i] < dist0[i]
            for Gi in @views G.rowval[nzrange(G, i)]
                Q[Gi] = true
            end
        end
    end
end

# function recontruct_path(D::BellmanFordMoore, source, receiver)
#     prev = D.prev
#     path = Int[receiver]
#     ipath = prev[receiver]
#     while ipath != source
#         push!(path, ipath)
#         # if !haskey(prev, ipath)
#         #     break
#         # end
#         ipath = prev[ipath]
#     end
#     push!(path, source)

#     return path
# end