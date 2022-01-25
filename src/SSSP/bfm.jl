function bfm(G::SparseMatrixCSC{Bool, M}, source::Int, gr, U::AbstractArray{T}) where {M,T}

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
    p = Vector{M}(undef, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    init_Q!(Q, G, e2n, source)

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)
        # @btime relax!($dist, $p, $dist0, $G, $Q, $e2n, $x, $z, $r, $U)
            
        # pop queue (serial-but-fast process)
        fillfalse!(Q)

        # update nodal queue (parallel process)
        update_Q!(Q, G, dist, dist0, e2n)
        # @btime update_Q!($Q, $G, $dist, $dist0, $e2n)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

function bfm(G::SparseMatrixCSC{Bool, M}, halo::Matrix, source::Int, gr, U::AbstractArray{T}) where {M,T}

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

    # allocate dictionary containing best previous nodes
    p = Vector{M}(undef, n)
    init_halo_path!(p, halo)
    
    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    init_Q!(Q, G, e2n, source)

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(Float64), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    to = TimerOutput()
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        # relax edges (parallel process)
        @timeit to "relax"  relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)
        # relax!(dist, p, dist0, G, Q, e2n, x, z, r, U, b)
        # @btime relax!($dist, $p, $dist0, $G, $Q, $e2n, $x, $z, $r, $U)
        # @btime relax2!($dist, $p, $dist0, $G, $Q, $e2n, $x, $z, $r, $U)
            
        @timeit to "halo update" update_halo!(p, dist, dist0, halo)
        
        # pop queue (serial-but-fast process)
        # fillfalse!(Q)
        @timeit to "fill"  fill!(Q, false)

        # update nodal queue (parallel process)
        @timeit to "update Q" update_Q!(Q, G, dist, dist0, e2n)
        # @btime update_Q!($Q, $G, $dist, $dist0, $e2n)

        # update old distance vector (TODO parallel version)
        @timeit to "copy"  copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return to
    # return BellmanFordMoore(p, dist)
end

function update_halo!(p, dist, dist0, halo)
    Threads.@threads for i in axes(halo,1)
        @inbounds if (dist[halo[i,1]] < dist0[halo[i,1]]) && (dist[halo[i,2]] > dist[halo[i,1]])
            dist[halo[i,2]] = dist[halo[i,1]]
            p[halo[i,2]] = p[halo[i,1]]
        end
    end
end

function init_halo_path!(p, halo)
    n = length(halo) ÷ 2
    @inbounds for i in 1:n
        p[halo[i, 2]] = halo[i,1]
        p[halo[i, 1]] = halo[i,2]
    end
end

sp_column(A::SparseMatrixCSC, I::T) where T<:Integer = @views A.rowval[nzrange(A, I)]

function init_Q!(Q::BitVector, G::SparseMatrixCSC{Bool, T}, e2n::Dict, source::Integer) where T
    for element in sp_column(G, source)
        for i in e2n[element]
            Q[i] = true
        end
    end
end

function update_Q!(Q::BitVector, G::SparseMatrixCSC, dist, dist0, e2n)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        if (dist[i] < Inf) && (dist[i] < dist0[i])
            for element in sp_column(G, i)
                for i in e2n[element]
                    # this line removes redundance and cuts 
                    # down the time of the function by ~half
                    Q[i] == true && continue
                    Q[i] = true
                end
            end
        end
    end
end


@inbounds function relax!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, e2n, x, z, r, U) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        _relax!(p, dist, dist0, G, i, e2n, x, z, r, U)
    end 
end

@inbounds function _relax!(p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, U::Matrix{T}) where T
   
    # read coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi, ri = x[i], z[i], r[i]
    Ui = (U[i, 1], U[i, 2])

    # queue to track redundant operations
    redundancyQ = Set{Int32}()
    
    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in  sp_column(G, i),  Gi in e2n[j]
        if Gi ∉ redundancyQ
            push!(redundancyQ, Gi)
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
                muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx]), dGi)
                # dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
            )

            # update distance and predecessor index if
            # it's smaller than the temptative distance
            if di > δ
                di = δ
                p[i] = Gi
            end
        end
    end

    # update distance
    dist[i] = di 
end

@inbounds function _relax!(p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, U::Vector{T}) where T
   
    # read coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi = x[i], z[i]
    Ui = U[i]
    
    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in  sp_column(G, i), Gi in e2n[j] 
        dGi = dist0[Gi]
        # temptative distance (ignore if it's ∞)
        δ = ifelse(
            dGi == typemax,
            typemax(T),
            muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui+U[Gi]), dGi)
            # dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
        )

        # update distance and predecessor index if
        # it's smaller than the temptative distance
        if di > δ
            di = δ
            p[i] = Gi
        end

        # dGi = dist0[Gi]
        # if dGi != typemax(T)
        #     δ = muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui+U[Gi]), dGi)

        #     # update distance and predecessor index if
        #     # it's smaller than the temptative distance
        #     if di > δ
        #         di = δ
        #         p[i] = Gi
        #     end
        # end
    end

    # update distance
    dist[i] = di 
end

@inbounds function _relax!(p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, interpolant::Interpolations.Extrapolation) 
   
    # read coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi = x[i], z[i]
    ri = r[i]
    # Ui = U[i]
    
    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in  sp_column(G, i), Gi in e2n[j] 
        dGi = dist0[Gi]
        # temptative distance (ignore if it's ∞)
        δ = ifelse(
            dGi == typemax,
            typemax(T),
            dGi + distance(xi, zi, x[Gi], z[Gi])/interpolant(0.5*(ri+r[Gi]))
            # muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui+U[Gi]), dGi)
            # dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
        )

        # update distance and predecessor index if
        # it's smaller than the temptative distance
        if di > δ
            di = δ
            p[i] = Gi
        end

        # dGi = dist0[Gi]
        # if dGi != typemax(T)
        #     δ = muladd(2, distance(xi, zi, x[Gi], z[Gi])/(Ui+U[Gi]), dGi)

        #     # update distance and predecessor index if
        #     # it's smaller than the temptative distance
        #     if di > δ
        #         di = δ
        #         p[i] = Gi
        #     end
        # end
    end

    # update distance
    dist[i] = di 
end


function bfmtest_bench(G::SparseMatrixCSC{Bool, M}, source::Int, gr, U::Matrix{T}) where {M,T}

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
    p = Vector{M}(undef, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    init_Q!(Q, G, e2n, source)

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    to = TimerOutput()
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        @timeit to "relax" relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)
        # @btime relax!($dist, $p, $dist0, $G, $Q, $e2n, $x, $z, $r, $U)
            
        # pop queue (serial-but-fast process)
        @timeit to "fill false" fillfalse!(Q)

        # update nodal queue (parallel process)
        @timeit to "update Q" update_Q!(Q, G, dist, dist0, e2n)
        # @btime update_Q!($Q, $G, $dist, $dist0, $e2n)

        # update old distance vector (TODO parallel version)
        @timeit to "copy dist" copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return to
end


function bfmtest_bench(G::SparseMatrixCSC{Bool, M}, halo::Matrix, source::Int, gr, U::AbstractArray{T}) where {M,T}

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

    # allocate dictionary containing best previous nodes
    p = Vector{M}(undef, n)
    init_halo_path!(p, halo)
    
    # priority queue containing NOT settled nodes
    Q = falses(n)
    # 1st frontier nodes are nodes adjacent to the source
    init_Q!(Q, G, e2n, source)

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    # covergence: if the queue is empty we are done
    to = TimerOutput()
    @inbounds while sum(Q) != 0
        # relax edges (parallel process)
        @timeit to "relax"  relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)
        # @btime relax!($dist, $p, $dist0, $G, $Q, $e2n, $x, $z, $r, $U)
            
        @timeit to "halo"  update_halo!(dist, dist0, halo)
        
        # pop queue (serial-but-fast process)
        @timeit to "fill"  fillfalse!(Q)

        # update nodal queue (parallel process)
        @timeit to "update"  update_Q!(Q, G, dist, dist0, e2n)
        # @btime update_Q!($Q, $G, $dist, $dist0, $e2n)

        # update old distance vector (TODO parallel version)
        @timeit to "copy"  copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return to
end