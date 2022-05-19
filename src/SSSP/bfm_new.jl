function bfmtest(G::SparseMatrixCSC{Bool,M}, source::Int, gr, U::Matrix{T}) where {M,T}

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
        it += 1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

function bfmtest_bench(
    G::SparseMatrixCSC{Bool,M}, source::Int, gr, U::Matrix{T}
) where {M,T}

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
        it += 1
    end

    println("Converged in $it iterations")

    return to
end

sp_colum(A::SparseMatrixCSC, I::T) where {T<:Integer} = @views A.rowval[nzrange(A, I)]

function init_Q!(
    Q::BitVector, G::SparseMatrixCSC{Bool,T}, e2n::Dict, source::Integer
) where {T}
    for element in sp_colum(G, source)
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
            for element in sp_colum(G, i)
                # for j in eachindex(element)
                for i in e2n[element]
                    # Q[e2n[element][j]] = true
                    Q[i] = true
                end
            end
        end
    end
end

@inline function relax!(
    dist::Vector{T},
    p::Vector,
    dist0,
    G::SparseMatrixCSC,
    Q::BitVector,
    e2n,
    x,
    z,
    r,
    U::Matrix,
) where {T}
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    # Threads.@threads 
    for i in findall(Q)
        _relax!(p, dist, dist0, G, i, e2n, x, z, r, U)
    end
end

function adjacents!(tmp, G, idx, e2n)
    N = 0
    indices = sp_colum(G, idx)
    @inbounds for j in indices
        for Gi in e2n[j]
            if Gi ∉ view(tmp, 1:(N + 1))
                N += 1
                tmp[N + 1] = Gi
            end
        end
    end
    return tmp[1] = N
end

@inbounds function _relax!(
    p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, U::Matrix{T}
) where {T}

    # read coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi, ri = x[i], z[i], r[i]
    Ui = (U[i, 1], U[i, 2])

    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in sp_colum(G, i), Gi in e2n[j]
        dGi = dist0[Gi]
        # i is the index of the ray-tail, Gi index of ray-head
        # branch-free arithmetic to check whether ray is coming from above or below
        # idx = 1 if ray is going downards, = 2 if going upwards
        head_idx = (ri > r[Gi]) + 1
        tail_idx = (head_idx == 1) + 1
        # temptative distance (ignore if it's ∞)
        δ = ifelse(
            dGi == typemax,
            typemax(T),
            muladd(
                2, distance(xi, zi, x[Gi], z[Gi]) / (Ui[tail_idx] + U[Gi, head_idx]), dGi
            ),
            # dGi + 2*distance(xi, zi, x[Gi], z[Gi])/(Ui[tail_idx]+U[Gi, head_idx])
        )

        # update distance and predecessor index if
        # it's smaller than the temptative distance
        if di > δ
            di = δ
            p[i] = Gi
        end
    end

    # update distance
    return dist[i] = di
end
