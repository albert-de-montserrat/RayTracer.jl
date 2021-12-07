@inbounds indices(Ga::SparseAdjencyList, source) = Ga.idx[source]:(Ga.idx[source+1]-1)

function bfm(Ga::SparseAdjencyList, source::Int, gr, U::Matrix{T}) where T

    # unpack coordinates
    x, z, r = gr.x, gr.z, gr.r

    # number of nodes in the graph
    n = length(x)

    # allocate dictionary containing best previous nodes
    p = fill(source, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    @inbounds for Gi in indices(Ga, source)
        Q[Ga.list[Gi]] = true
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
        relax_bfm!(dist, p, dist0, Ga, Q, x, z, r, U)
            
        # pop queue 
        fillfalse!(Q)

        # update nodal queue (parallel process)
        _update_bfm!(Q, Ga, dist, dist0)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

@inline function relax_bfm!(dist::Vector{T}, p::Vector, dist0, Ga::SparseAdjencyList, Q::BitVector, x, z, r, U::Matrix) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        _relax_bfm!(p, dist, dist0, Ga, i, x, z, r, U)
    end
end

@inbounds function _relax_bfm!(p::Vector, dist, dist0, Ga::SparseAdjencyList, i, x, z, r, U::Matrix{T}) where T
   
    # cache coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi, ri = x[i], z[i], r[i]
    Ui = (U[i, 1], U[i, 2])

    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for idx in indices(Ga, i)
        # current node
        Gi = Ga.list[idx]
        # previous distance
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
            dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
            # muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx]), dGi)
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
end

function _update_bfm!(Q::BitVector, Ga::SparseAdjencyList, dist, dist0)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        if dist[i] < dist0[i]
            for idx in indices(Ga, i)
                # current node
                Q[Ga.list[idx]] = true
            end
        end
    end
end

@inbounds function get_Qi(gr)
    inode = 60

    for i in  @views gr.neighbours[inode]
        for j in gr.e2n[i]
            Qi = j 
        end
    end
end