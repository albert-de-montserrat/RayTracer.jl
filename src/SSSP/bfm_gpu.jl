# Single-node parallel implementation of the  
# Bellman-Ford-Moore (BFM) SSSP algorithm
# implemented with a priority queue and predecessor
# approach (better suited for multi-threading). 
# The graph of nodes V and edges E is represented
# by an adjacency list G
using CUDA.CUSPARSE

struct CuGraph{T}
    K::T
    n1::T
    n2::T
end

struct CuMesh2D{T}
    x::T
    z::T
    U::T
end

function e2n2gpu(e2n::Dict{T,Vector{T}}) where {T}
    nonz = sum(length(e) for (_, e) in e2n)
    I, J, V = Vector{T}(undef, nonz), Vector{T}(undef, nonz), Vector{Bool}(undef, nonz)
    c = 0
    for (iel, element) in e2n
        @inbounds for node in element
            c += 1
            I[c] = node
            J[c] = iel
            V[c] = true
        end
    end

    return S = CuSparseMatrixCSC(sparse(I, J, V))
end

function _gpu_relaxation_BFM!(Q, K, p, dist, dist0, n1, n2, x, z, U)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    T = typemax(Float32)

    if index < length(Q)
        @inbounds if Q[index]
            di::Float64 = dist0[index]
            # xi::Float64, zi::Float64, Ui::Float64 = x[index], z[index], U[index]
            # TODO cache out xj yj zj ?
            for i in n1[index]:n2[index]
                Gi = K[i]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                    dist0[Gi] == T,
                    T,
                    dist0[Gi] +
                    2 * distance(x[index], z[index], x[Gi], z[Gi]) / (U[index] + U[Gi]),
                )
                # update distance and predecessor index 
                # if it's smaller than the temptative distance
                if di > δ
                    di = δ
                    p[index] = Gi
                end
            end

            # update distance
            dist[index] = di
        end
    end

    return nothing
end

function relaxation_BFM!(
    Q::CuArray{Bool},
    dist::CuArray,
    dist0::CuArray,
    p::CuArray,
    mesh_d::CuMesh2D,
    graph_d::CuGraph,
)
    # unpack graph arrays
    K, n1, n2 = graph_d.K, graph_d.n1, graph_d.n2

    # unpack coordinates and velocity
    x, z, U = mesh_d.x, mesh_d.z, mesh_d.U

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_relaxation_BFM!(
            Q, K, p, dist, dist0, n1, n2, x, z, U
        )
    end
end

function _gpu_update_bfm!(Q, K, n1, n2, dist, dist0)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index < length(Q)
        @inbounds if dist[index] < dist0[index]
            for i in n1[index]:n2[index]
                Q[K[i]] = true
            end
        end
    end

    return nothing
end

function update_bfm!(Q::CuArray{Bool}, dist::CuArray, dist0::CuArray, graph_d::CuGraph)
    # unpack graph arrays
    K, n1, n2 = graph_d.K, graph_d.n1, graph_d.n2

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_update_bfm!(Q, K, n1, n2, dist, dist0)
    end
end

function sparse2gpu(K::SparseMatrixCSC)
    # Setup Graph arrays
    K_d = CuVector(Int32.(K.rowval)) # sparse nodal connectivity
    # indices of K corresponding to nodes connected to the i-th node
    nz1 = Vector{Int32}(undef, K.n)
    nz2 = similar(nz1)
    @inbounds for j in 1:(K.n)
        r = nzrange(K, j)
        # a, b = extrema(r.start, r.stop) # indices can be inverted e.g. imax:imin
        a = min(r.start, r.stop) # indices can be inverted e.g. imax:imin
        b = max(r.start, r.stop) # indices can be inverted e.g. imax:imin
        nz1[j] = a
        nz2[j] = b
    end
    n1_d, n2_d = CuArray(nz1), CuArray(nz2)

    return graph_d = CuGraph(K_d, n1_d, n2_d)
end

function list2gpu(K::Dict)
    nonz = sum(length(e) for (_, e) in K)
    n = length(K)
    # Setup Graph arrays
    KK = Vector{Int32}(undef, nonz) # sparse nodal connectivity
    # indices of K corresponding to nodes connected to the i-th node
    nz1 = zeros(Int32, n)
    nz2 = zeros(Int32, n)
    # @inbounds 
    for i in 1:n
        el = K[i]
        if i == 1
            nz1[i] = 1
            nz2[i] = length(el)
        else
            nz1[i] = nz2[i - 1] + 1
            nz2[i] = nz2[i - 1] + length(el)
        end

        for (k, j) in enumerate(nz1[i]:nz2[i])
            KK[j] = el[k]
        end
    end
    K_d, n1_d, n2_d = CuArray(KK), CuArray(nz1), CuArray(nz2)

    return graph_d = CuGraph(K_d, n1_d, n2_d)
end

function move2device(K, U, gr, source)

    # move velocity field to device
    U_d = CuArray(Float32.(U))

    # move spatial coordinates arrays to device (TODO specialize for lazy grids)
    x_d = CuArray(Float32.(gr.x))
    z_d = CuArray(Float32.(gr.z))

    mesh_d = CuMesh2D(x_d, z_d, U_d)

    element_connectivity_d = sparse2gpu(K)

    e2n_d = list2gpu(gr.e2n)

    # number of nodes in the graph
    n = length(U)

    # allocate dictionary containing best previous nodes
    p = CUDA.zeros(Int32, n)

    # priority queue containing NOT settled nodes
    Q = CuVector(fill(false, n))

    # # 1st frontier: nodes adjacent to the source
    # isource = K.rowval[nzrange(K, source)]
    # Q[isource] .= true

    # initialise all distances as infinity and zero at the source node 
    tmp = fill(Inf, n)
    tmp[source] = 0
    dist = CuArray{Float32}(tmp)
    dist0 = deepcopy(dist)

    return element_connectivity_d, e2n_d, mesh_d, Q, dist, dist0, p
end

function bfm_gpu(
    G::SparseMatrixCSC{Bool,M}, halo::Matrix, source::Int, gr, U::AbstractArray{T}
) where {M,T}
    element_connectivity_d, e2n_d, mesh_d, Q, dist, dist0, p = move2device(G, U, gr, source)

    # allocate dictionary containing best previous nodes
    halo_d = CuArray(halo)
    init_halo!(p, halo_d)

    # 1st frontier nodes are nodes adjacent to the source
    init_Q!(Q, element_connectivity_d, e2n_d, source)

    # main loop
    it = 1

    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        # relax edges (parallel process)         
        relaxation_BFM2!(Q, dist, dist0, p, mesh_d, element_connectivity_d, e2n_d)

        # update_halo!(p, dist, dist0, halo_d)
        update_halo2!(p, dist, dist0, halo_d)

        # pop queue (serial-but-fast process)
        CUDA.fill!(Q, false)

        # update nodal queue (parallel process)
        update_Q!(Q, dist, dist0, element_connectivity_d, e2n_d)
        # @btime update_Q!($Q, $G, $dist, $dist0, $e2n)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it += 1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

# function init_halo_path!(p, halo)
#     n = length(halo) ÷ 2
#     @inbounds for i in 1:n
#         p[halo[i, 2]] = halo[i,1]
#         p[halo[i, 1]] = halo[i,2]
#     end
# end

function _init_halo!(p, halo_d, n)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index ≤ n
        h1::Int = halo_d[index, 1]
        h2::Int = halo_d[index, 2]
        p[h2] = h1
        p[h1] = h2
    end

    return nothing
end

function init_halo!(p::CuArray, halo_d::CuArray)
    n = size(halo_d, 1) ÷ 2
    nt = 256
    numblocks = ceil(Int, size(halo_d, 1) / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _init_halo!(p, halo_d, n + 1)
    end
end

function _update_halo!(p, dist, dist0, halo_d)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index ≤ size(halo_d, 1)
        @inbounds h1::Int = halo_d[index, 1]
        @inbounds h2::Int = halo_d[index, 2]
        @inbounds if (dist[h1] < dist0[h1]) && (dist[h2] > dist[h1])
            dist[h2] = dist[h1]
            p[h2] = p[h1]
        end
    end

    return nothing
end

function update_halo!(p::CuVector, dist::CuVector, dist0::CuVector, halo_d::CuMatrix)
    n = size(halo_d, 1) ÷ 2
    nt = 256
    numblocks = ceil(Int, size(halo_d, 1) / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _update_halo!(p, dist, dist0, halo_d)
    end
end

function _update_halo2!(p, dist, dist0, halo_d)

    # Set up shared memory cache for this current block.
    dh1 = @cuDynamicSharedMem(Float32, 256)
    dh2 = @cuDynamicSharedMem(Float32, 256)
    d0h2 = @cuDynamicSharedMem(Float32, 256)

    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # tid = threadIdx().x

    @inbounds if index ≤ size(halo_d, 1)
        h1::Int = halo_d[index, 1]
        h2::Int = halo_d[index, 2]

        dh1[tid] = dist[h1]
        dh2[tid] = dist[h2]
        d0h2[tid] = dist0[h2]
        synchronize()

        # dh1::Float32 = dist[h1] 
        # dh2::Float32 = dist[h2] 
        # d0h1::Float32 = dist0[h1] 

        if (dh1[tid] < d0h1[tid]) && (dh2[tid] > dh1[tid])
            dist[h2] = dh1[tid]
            p[h2] = p[h1]
        end
    end

    return nothing
end

function update_halo2!(
    p::CuVector, dist::CuVector{T}, dist0::CuVector{T}, halo_d::CuMatrix
) where {T}
    nt = 256
    mem = (3 * nt * sizeof(T))

    numblocks = ceil(Int, size(halo_d, 1) / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks shmem = mem _update_halo!(
            p, dist, dist0, halo_d
        )
    end
end

function _init_Q!(Q, K, e2n, n11, n21, n12, n22, source)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index == source
        @inbounds for i in n11[index]:n21[index]
            element::Int32 = K[i]
            for j in n12[element]:n22[element]
                Q[e2n[j]] = true
            end
        end
    end

    return nothing
end

function init_Q!(Q::CuArray{Bool}, element_connectivity_d::CuGraph, e2n_d::CuGraph, source)
    # unpack graph arrays
    K, n11, n21 = element_connectivity_d.K,
    element_connectivity_d.n1,
    element_connectivity_d.n2
    e2n, n12, n22 = e2n_d.K, e2n_d.n1, e2n_d.n2

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _init_Q!(
            Q, K, e2n, n11, n21, n12, n22, source
        )
    end
end

function _update_Q!(Q, K, e2n, dist, dist0, n11, n21, n12, n22)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    inf = typemax(eltype(dist))
    if index ≤ length(Q)
        @inbounds if (dist[index] < inf) && (dist[index] < dist0[index])
            for i in n11[index]:n21[index]
                element::Int32 = K[i]
                for j in n12[element]:n22[element]
                    node::Int32 = e2n[j]
                    Q[node] == true && continue
                    Q[node] = true
                end
            end
        end
    end

    return nothing
end

function update_Q!(
    Q::CuVector{Bool},
    dist::CuVector,
    dist0::CuVector,
    element_connectivity_d::CuGraph,
    e2n_d::CuGraph,
)
    # unpack graph arrays
    K, n11, n21 = element_connectivity_d.K,
    element_connectivity_d.n1,
    element_connectivity_d.n2
    e2n, n12, n22 = e2n_d.K, e2n_d.n1, e2n_d.n2

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _update_Q!(
            Q, K, e2n, dist, dist0, n11, n21, n12, n22
        )
    end
end

function _update_Q2!(Q, K, e2n, dist, dist0, n11, n21, n12, n22)

    # Set up shared memory cache for this current block.
    element_cache = @cuDynamicSharedMem(Float32, 256)

    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x
    inf = typemax(Float32)

    if index ≤ length(Q)
        @inbounds if (dist[index] < inf) && (dist[index] < dist0[index])
            for i in n11[index]:n21[index]
                element::Int32 = K[i]
                for j in n12[element]:n22[element]
                    Q[e2n[j]] == true && continue
                    Q[e2n[j]] = true
                end
            end
        end
    end

    return nothing
end

function update_Q2!(
    Q::CuVector{Bool},
    dist::CuVector,
    dist0::CuVector,
    element_connectivity_d::CuGraph,
    e2n_d::CuGraph,
)
    # unpack graph arrays
    K, n11, n21 = element_connectivity_d.K,
    element_connectivity_d.n1,
    element_connectivity_d.n2
    e2n, n12, n22 = e2n_d.K, e2n_d.n1, e2n_d.n2

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    mem = sum(a[i] for i in (length(a) - nt):length(a)) * sizeof(Float32)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks shmem = mem _update_Q!(
            Q, K, e2n, dist, dist0, n11, n21, n12, n22
        )
    end
end

function _gpu_relaxation_BFM2!(Q, K, e2n, p, dist, dist0, n11, n21, n12, n22, x, z, U)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    T = typemax(Float32)

    if index ≤ length(Q)
        @inbounds if Q[index]

            # read coordinates, velocity and distance of frontier node
            di::Float32 = dist0[index]
            xi::Float32, zi::Float32 = x[index], z[index]
            Ui::Float32 = U[index]

            # xi::Float64, zi::Float64, Ui::Float64 = x[index], z[index], U[index]
            for i in n11[index]:n21[index]
                element::Int32 = K[i]
                for j in n12[element]:n22[element]
                    Gi = e2n[j]
                    # temptative distance (ignore if it's ∞)
                    δ = ifelse(
                        dist0[Gi] == T,
                        T,
                        dist0[Gi] + 2 * distance(xi, zi, x[Gi], z[Gi]) / (Ui + U[Gi]),
                    )

                    # update distance and predecessor index 
                    # if it's smaller than the temptative distance
                    if di > δ
                        di = δ
                        p[index] = Gi
                    end
                end
            end

            # update distance
            dist[index] = di
        end
    end

    return nothing
end

function relaxation_BFM2!(
    Q::CuArray{Bool},
    dist::CuArray,
    dist0::CuArray,
    p::CuArray,
    mesh_d::CuMesh2D,
    element_connectivity_d::CuGraph,
    e2n_d::CuGraph,
)

    # unpack graph arrays
    K, n11, n21 = element_connectivity_d.K,
    element_connectivity_d.n1,
    element_connectivity_d.n2
    e2n, n12, n22 = e2n_d.K, e2n_d.n1, e2n_d.n2

    # unpack coordinates and velocity
    (; x, z, U) = mesh_d

    nt = 256
    numblocks = ceil(Int, length(Q) / nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_relaxation_BFM2!(
            Q, K, e2n, p, dist, dist0, n11, n21, n12, n22, x, z, U
        )
    end
end
