# Single-node parallel implementation of the  
# Bellman-Ford-Moore (BFM) SSSP algorithm
# implemented with a priority queue and predecessor
# approach (better suited for multi-threading). 
# The graph of nodes V and edges E is represented
# by an adjacency list G

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

# specialized implementation for on-the-fly weights
function bfm_gpu(Gsp::SparseMatrixCSC, source::Int, gr, U::Vector{T}) where {M,T}

    # move arrays to GPU
    graph_d, mesh_d = move2device(Gsp, U, gr)

    # number of nodes in the graph
    n = length(U)

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
    p = CUDA.zeros(Int32, n)

    # priority queue containing NOT settled nodes
    Q = CUDA.fill(false,n)
    # 1st frontier: nodes adjacent to the source
    isource = Gsp.rowval[nzrange(Gsp, source)]
    Q[isource] .= true

    # initialise all distances as infinity and zero at the source node 
    tmp = fill(typemax(T), n)
    tmp[source] = zero(T)
    dist = CuArray(tmp)
    tmp = nothing
    # dist = CUDA.fill(typemax(T), n)
    # dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main loop
    it = 1
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        relaxation_BFM!(Q, dist, dist0, p, mesh_d, graph_d)
        
        # pop queue (serial-but-fast process)
        CUDA.fill!(Q, false)

        # update nodal queue (parallel process)
        update_bfm!(Q, dist, dist0, graph_d)
        # @btime update_bfm!($Q, $dist, $dist0, $graph_d)

       # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        it+=1
    end

    println("Converged in $it iterations")
    p_cpu = Array(p)
    dist_cpu = Array(dist)

    return BellmanFordMoore(p_cpu, dist_cpu)
end

function _gpu_relaxation_BFM!(Q, K, p, dist, dist0, n1, n2, x, z, U)

    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    T = Float32

    if index < length(Q)

        if Q[index]

            di = dist0[index]

            # TODO cache out xj yj zj ?
            for i in n1[index]:n2[index]
                Gi = K[i]
                # temptative distance (ignore if it's ∞)
                δ = ifelse(
                   dist0[Gi] == typemax(T),
                   typemax(T),
                   dist0[Gi] + 2*distance(x[index], z[index], x[Gi], z[Gi])/abs(U[index]+U[Gi])
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

    return
end

function relaxation_BFM!(Q::CuArray{Bool}, dist::CuArray, dist0::CuArray, p::CuArray, mesh_d::CuMesh2D, graph_d::CuGraph)
    # unpack graph arrays
    K, n1, n2 = graph_d.K, graph_d.n1, graph_d.n2

    # unpack coordinates and velocity
    x, z, U = mesh_d.x, mesh_d.z, mesh_d.U

    nt = 256
    numblocks = ceil(Int, length(Q)/nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_relaxation_BFM!(Q, K, p, dist, dist0, n1, n2, x, z, U)
    end
end

@kernel function potato!(Q, K, n1, n2, dist, dist0)
    index = @index(Global)
    
    if index < length(Q)
        @inbounds if dist[index] < dist0[index]        
            for i in n1[index]:n2[index]
                Q[K[i]] = true
            end
        end
    end

end

function foo!(Q, K, n1, n2, dist, dist0)
    kernel! = potato!(CUDADevice(), 256)
    event = kernel!(Q, K, n1, n2, dist, dist0, ndrange=length(Q))
    wait(event)
end

function _gpu_update_bfm!(Q, K, n1, n2, dist, dist0)

    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # T = Float32
    if index < length(Q)
        
        @inbounds if dist[index] < dist0[index]
            
            for i in n1[index]:n2[index]
                Q[K[i]] = true
            end

        end

    end

    return 
end

function update_bfm!(Q::CuArray{Bool}, dist::CuArray, dist0::CuArray, graph_d::CuGraph)
    # unpack graph arrays
    K, n1, n2 = graph_d.K, graph_d.n1, graph_d.n2

    nt = 256
    numblocks = ceil(Int, length(Q)/nt)

    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_update_bfm!(Q, K, n1, n2, dist, dist0)
    end
end

function move2device(K, U, gr)

    # move velocity field to device
    U_d = CuArray(Float32.(U))

    # move spatial coordinates arrays to device (TODO specialize for lazy grids)
    x_d = CuArray(Float32.(gr.x))
    z_d = CuArray(Float32.(gr.z))

    # Setup Graph arrays
    K_d = CuVector(Int32.(K.rowval)) # sparse nodal connectivity
    # indices of K corresponding to nodes connected to the i-th node
    nz1 = Vector{Int32}(undef, K.n)
    nz2 = similar(nz1)

    @inbounds for j in 1:K.n
        r = nzrange(K, j)
        a = min(r.start, r.stop) # indices can be inverted e.g. imax:imin
        b = max(r.start, r.stop) # indices can be inverted e.g. imax:imin
        nz1[j] = a
        nz2[j] = b
    end
    n1_d, n2_d = CuArray(nz1), CuArray(nz2)

    graph_d = CuGraph(K_d, n1_d, n2_d)
    mesh_d = CuMesh2D(x_d, z_d, U_d)

    return graph_d, mesh_d

end