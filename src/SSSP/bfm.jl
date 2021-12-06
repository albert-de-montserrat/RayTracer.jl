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

# specialized implementation for on-the-fly weights
function bfm(G::Dict{M, Set{M}}, source::Int, gr, U::Matrix{T}) where {M,T}

    # unpack coordinates
    x, z, r = gr.x, gr.z, gr.r

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
        _relax_bfm!(dist, p, dist0, G, Q, x, z, r, U)
            
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

function bfm2(Gsp::SparseMatrixCSC, source::Int, gr, U::Vector{T}, partition, interpolant) where T

    # unpack partition
    ID = partition.id
    rboundaries = partition.rboundaries
    nlayers = partition.nlayers
    nboundaries = partition.nboundaries
    layers = partition.layers
    boundaries = partition.boundaries
    iterator = partition.iterator
 
    boundary_dict = Dict(a => b for (a,b) in zip( partition.boundaries,  partition.rboundaries))

    # find ids of boundary nodes
    boundary_nodes =  reduce(
        vcat,
        [findall(round.(gr.r, digits=2) .== b) for b in partition.rboundaries]
    )
    
    # unpack coordinates
    x, z = gr.x, gr.z

    # number of nodes in the graph
    n = length(x)

    # allocate dictionary containing best previous nodes
    p = Vector{Int}(undef, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    @inbounds for Gi in @views Gsp.rowval[nzrange(Gsp, source)]
        Q[Gi] = true
    end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    # covergence: if the queue is empty we are done
    while sum(Q) != 0
        
        # relax edges (parallel process)
        _relax_bfm!(dist, p, dist0, Gsp, Q, x, z, U)
            
        # pop queue 
        fillfalse!(Q)

        # update nodal queue (parallel process)
        _update_bfm!(Q, Gsp, dist, dist0)

        # Threads.@threads for i in eachindex(Q)
        #     if dist[i] < dist0[i]
        #         for Gi in @views G.rowval[nzrange(G, i)]
        #             Q[Gi] = true
        #         end
        #     end
        # end

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        for idx in boundary_nodes
            if Q[idx] == true
                U[idx] = interpolant(gr.r[idx] + 1)
            else
                # if (dist[idx] != typemax(T)) 
                U[idx] = interpolant(gr.r[idx] - 1)
            end
        end

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
    n = length(x)

    # allocate dictionary containing best previous nodes
    p = fill(source, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    @inbounds for Gi in @views Gsp.rowval[nzrange(Gsp, source)]
        Q[Gi] = true
    end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    # covergence: if the queue is empty we are done
    while sum(Q) != 0
        
        # relax edges (parallel process)
        _relax_bfm!(dist, p, dist0, Gsp, Q, x, z, U)
            
        # pop queue 
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

function bfm(Gsp::SparseMatrixCSC, source::Int, gr, U::Matrix{T}) where T

    # unpack coordinates
    x, z, r = gr.x, gr.z, gr.r

    # number of nodes in the graph
    n = length(x)

    # allocate dictionary containing best previous nodes
    p = fill(source, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    @inbounds for Gi in @views Gsp.rowval[nzrange(Gsp, source)]
        Q[Gi] = true
    end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n) 
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    # covergence: if the queue is empty we are done
    while sum(Q) != 0
        
        # relax edges (parallel process)
        _relax_bfm!(dist, p, dist0, Gsp, Q, x, z, r, U)
            
        # pop queue 
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

function foo!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, x, z, U) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching

    idx = findall(Q)
    xx = @views x[idx]
    zz = @views z[idx]
    UU = @views U[idx]

    degrees = [length(nzrange(G, i)) for i in idx] 
    max_degree = maximum(degrees)
    xbuffer = fill(NaN, max_degree,length(degrees))
    zbuffer = fill(NaN, max_degree, length(degrees))
    Ubuffer = fill(NaN, max_degree, length(degrees))
    d0buffer = fill(NaN, max_degree, length(degrees))

    # Threads.@threads 
    for ii in eachindex(idx)
        # iterate over adjacent nodes to find the the one with 
        # the mininum distance to current node
        for (j, Gi) in enumerate(@views G.rowval[nzrange(G, idx[ii])])
            xbuffer[j, ii] = x[Gi]
            zbuffer[j, ii] = z[Gi]
            Ubuffer[j, ii] = U[Gi]
            d0buffer[j, ii] = dist0[Gi]
        end
    end

    # Threads.@threads 
    for ii in eachindex(idx)
        
            # cache coordinates, velocity and distance of frontier node
            di = dist0[ii]

            # iterate over adjacent nodes to find the the one with 
            # the mininum distance to current node
            for j in 1:degrees[ii]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                    d0buffer[j, ii] == typemax(T),
                    typemax(T),
                    d0buffer[j, ii] + 2*distance(xx[ii], zz[ii], xbuffer[j, ii], zbuffer[j, ii])/abs(UU[ii]+Ubuffer[j, ii])
                )

                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if di > δ
                    di = δ
                    p[idx[ii]] = j
                end
            end

            # update distance
            dist[ii] = di 
    end
end

function _relax_bfm!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, x, z, U) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        
        # @inbounds if Q[i]
            # cache coordinates, velocity and distance of frontier node
            di = dist0[i]

            # iterate over adjacent nodes to find the the one with 
            # the mininum distance to current node
            for Gi in @views G.rowval[nzrange(G, i)]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                    dist0[Gi] == typemax(T),
                    typemax(T),
                    dist0[Gi] + 2*distance(x[i], z[i], x[Gi], z[Gi])/(U[i]+U[Gi])
                )

                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if di > δ
                    di = δ
                    p[i] = Gi
                end
            end

            # update distance
            dist[i] = di 
        # end
    end
end

@inline function relax_bfm!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, x, z, r, U::Matrix) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    # Threads.@threads 
    for i in findall(Q)
        
        # @inbounds if Q[i]
            # cache coordinates, velocity and distance of frontier node
            di = dist0[i]

            # iterate over adjacent nodes to find the the one with 
            # the mininum distance to current node
            for Gi in @views G.rowval[nzrange(G, i)]
                dGi = dist0[Gi]                
                # i is the index of the ray-tail, Gi index of ray-head
                # branch-free arithmetic to check whether ray is coming from above or below
                # idx = 1 if ray is going downards, = 2 if going upwards
                head_idx = (r[i] > r[Gi]) + 1
                tail_idx = (head_idx==1) + 1
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                    dGi == typemax(T),
                    typemax(T),
                    dGi + 2*distance(x[i], z[i], x[Gi], z[Gi])/(U[i, tail_idx]+U[Gi, head_idx])
                )

                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if dist0[i] > δ
                    dist0[i] = δ
                    p[i] = Gi
                end
            end

            # update distance
            dist[i] = di 
        # end
    end
end

@noinline function _relax_bfm!(p::Vector, dist, dist0, G::SparseMatrixCSC, i, x, z, r, U::Matrix{T}) where T
   
    # cache coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi, ri = x[i], z[i], r[i]
    Ui = (U[i, 1], U[i, 2])

    # # UGi = @SMatrix [U[Gi, :] for Gi in @views G.rowval[nzrange(G, i)]]
    # idx = @views G.rowval[nzrange(G, i)]
    # xGi = @views x[idx]
    # zGi = @views z[idx]
    # rGi = @views r[idx]
    # dGi = @views dist0[idx]
    # UGi = @views U[idx, :]

    # @inbounds for i in eachindex(idx)
    #     # i is the index of the ray-tail, Gi index of ray-head
    #     # branch-free arithmetic to check whether ray is coming from above or below
    #     # idx = 1 if ray is going downards, = 2 if going upwards
    #     head_idx = (ri > rGi[i]) + 1
    #     tail_idx = (head_idx==1) + 1
    #     # temptative distance (ignore if it's ∞)
    #     δ = ifelse(
    #         dGi[i] == typemax(T),
    #         typemax(T),
    #         dGi[i] + 2*distance(xi, zi, xGi[i], zGi[i])/(Ui[tail_idx]+UGi[i, head_idx])
    #     )

    #     # update distance and predecessor index 
    #     # if it's smaller than the temptative distance
    #     if di > δ
    #         di = δ
    #         p[i] = Gi
    #     end
    # end

    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    @inbounds for Gi in @views G.rowval[nzrange(G, i)]
        dGi = dist0[Gi]                
        # i is the index of the ray-tail, Gi index of ray-head
        # branch-free arithmetic to check whether ray is coming from above or below
        # idx = 1 if ray is going downards, = 2 if going upwards
        head_idx = (ri > r[Gi]) + 1
        tail_idx = (head_idx==1) + 1
        # temptative distance (ignore if it's ∞)
        δ = ifelse(
            dGi == typemax,
            typemax(T),
            # dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
            muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx]), dGi)
        )

        # update distance and predecessor index 
        # if it's smaller than the temptative distance
        if di > δ
            di = δ
            p[i] = Gi
        end
    end

    # # update distance
    # dist[i] = di 
end

@btime _relax_bfm!($p, $dist, $dist0, $Gsp, $120, $x, $z, $r, $Vp);
@code_warntype _relax_bfm!(p, dist, dist0, Gsp, 120, x, z, r, Vp);

@btime _relax_bfm!($dist, $p, $dist0, $Gsp, $Q, $x, $z, $r,$Vp)


# struct Queue{T, I}
#     indices::T
#     n::I
# end

# @inline function indices_Q(idxQ, Q)
#     # c = count(Q)
#     n = 0
#     it = 0
#     i1 = findfirst(Q)
#     i2 = findlast(Q)
#     for it in i1:i2
#         @inbounds if Q[it]
#             n += 1 
#             idxQ[n] = it
#         end
#         # n == c && break
#     end
#     return idxQ, n
# end

# @code_warntype_relax_bfm!(dist, p, dist0, Gsp, Q, x, z, r,Vp)

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
    # iterate over queue. Unfortunately @threads can't iterate 
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
                dist0[Gi] + 2*distance(xi, zi, x[Gi], z[Gi])/(U[i]+Ui)
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
    for i in 1:length(G)
        @inbounds if dist[i] < dist0[i]
            union!(Q, G[i])
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
        if dist[i] < dist0[i]
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