struct Dijsktra{T, M}
    prev::Dict{T, T}
    dist::M
end

struct CuDijsktra{T, M}
    prev::T
    dist::M
end

struct CuGraph{T}
    K::T
    n1::T
    n2::T
end

struct CuMesh{T}
    x::T
    y::T
    z::T
    U::T
end

const ∞ = Inf

function dijsktra(G, source, gr, U, fw::Function)

    n = length(G)
    # p = fill(source, n) # best previous nodes
    p = Dict{Int, Int}() # best previous nodes
    dist = fill(Inf, n) # distance from best previous node
    dist[source] = 0
    # dist = [fw(gr[source], gr[i]) for i in 1:n]
    Q = Set{Int}() # queue
    visited = Set{Int}() # visited nodes
    push!(Q, source)
    nit = 0
    while !isempty(Q)
        Qi = min_distance(Q, dist) # index from priority queue with min distance
        gri = gr[Qi]
        Ui = U[Qi]
        di = dist[Qi]
        for i in G[Qi]
            i ∈ visited && continue # skips lines below if i ∈ visited = true
            tmp_distance = di + fw(gri, gr[i])/abs(U[i]+Ui)*0.5
            if tmp_distance < dist[i]
                p[i] = Qi
                dist[i] = tmp_distance
                push!(Q, i)
            end
        end
        delete!(Q, Qi)
        push!(visited, Qi)
        nit+=1
    end

    return Dijsktra(p, dist)
end

function min_distance(Q, dist)
    d = Inf
    idx = -1
    for Qi in Q
        @inbounds di = dist[Qi]
        @inbounds if di < d
            d = di
            idx = Qi
        end
    end
    return idx
end

function recontruct_path(D, source, target)
    prev = D.prev
    path = Int[target]
    ipath = prev[target]
    while ipath != source
        push!(path, ipath)
        if !haskey(prev, ipath)
            break
        end
        ipath = prev[ipath]
    end
    push!(path, source)

    return path
end

function recontruct_forward_path(D, source, target)
    prev = D.prev
    path = Int[]
    ipath = prev[target]
    while ipath != source
        push!(path, ipath)
        if !haskey(prev, ipath)
            break
        end
        ipath = prev[ipath]
    end

    return path
end

function dijsktra_parallel(G, source, gr, U, fw::Function)

    n = length(G)
    p = Dict{Int, Int}() # best previous nodes
    dist = fill(∞, n) # distance from best previous node
    dist[source] = 0
    # dist = [fw(gr[source], gr[i]) for i in 1:n]
    Q = fill(true, n) # Settled queue
    Q[1] = false
    F = fill(false, n) # Frontier queue
    F[1] = true
    Δ = 0.0
    it = 0
    # TODO: swtich @threads -> @spawn, highly unbalanced jobs
    while sum(Q) != 0
        relaxation!(Q, F, G, dist, p, gr, U, fw)
        Δ = min_distance(Q, dist) # threaded version only worth it for very large problems
        update!(F, Q, dist, Δ)
        it+=1
    end
    
    println("Converged in $it iterations")

    return Dijsktra(p, dist)
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

function _update!(F, Q, dist, Δ)
    F = false
    b = (Q == true) && (dist ≤ Δ)
    b && (Q = false) && (F = true)
    # return F, Q
end

function relaxation!(Q, F, G, dist, p, gr, U, fw::Function)
    Threads.@threads for i in eachindex(F)
        @inbounds if F[i]
            for j in G[i]
                Q[j] != true && continue
                tmp_dist = dist[i] + fw(gr[i], gr[j])/abs(U[j]+U[i])*0.5
                if dist[j] > tmp_dist
                    dist[j] = tmp_dist
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

function _gpu_update!(F, Q, dist, Δ)
    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if index < length(F)
        F[index] = false
        if (Q[i] == true) && (dist[i] ≤ Δ)
            Q[index] = false
            F[index] = true
        end
    end
    return 
end

function gpu_update!(F::CuArray{Bool}, Q::CuArray{Bool}, dist::CuArray{T}, Δ) where T
    nt = 512
    numblocks = ceil(Int, length(F)/nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_update!(F, Q, dist, Δ)
    end
end

function _gpu_relaxation!(Q, F, K, dist, n1, n2, x, z, U)
    index = blockIdx().x * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    # index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for i = index:stride:length(F)
        @inbounds if F[i]
            for j in n1[i]:n2[i]
                if Q[K[j]]
                    CUDA.@atomic dist[j] = min(
                        dist[j],
                        dist[i] + distance2D(x[i], z[i], x[j], z[j])*abs(U[j]-U[i])*0.5
                    )
                end
            end
        end
    end
end

function gpu_relaxation!(Qd, Fd, Kd, dd, id1, id2, xd, zd, Ud)
    nt = 256
    numblocks = ceil(Int, (length(F) + nt -1)/nt)
    # numblocks = ceil(Int, length(F)/nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_relaxation!(Qd, Fd, Kd, dd, id1, id2, xd, zd, Ud)
    end
end

function _gpu_min_distance!(Q, dist, out)
    nthreads = blockDim().x # threads available per block
    sdata = @cuDynamicSharedMem(Float64, nthreads) # set up shared memory cache for this current block

    index = (blockIdx().x-1) * 2*blockDim().x + threadIdx().x
    tid = threadIdx().x -1 # cache index 
    I = index + blockDim().x

    # TODO need to handle tail

    @inbounds d1::Float64 = Q[index] == true ? dist[index] : Inf
    @inbounds d2::Float64 = Q[I] == true ? dist[index] : Inf
    @inbounds sdata[tid+1] = min(d1, d2)
    
    sync_threads() # synchronize and restart threads
    
    i::Int = blockDim().x ÷ 2
    while i != 0 # somehow while is much faster here
        if tid < i
            @inbounds sdata[tid+1] = min(sdata[tid+1], sdata[tid+i+1])
        end
        sync_threads() # synchronize and restart threads
        i = i ÷ 2
    end

    # sdata[1] now contains the sum of vector dot product calculations done in
    # this block
    if tid == 1
        @inbounds out[blockIdx().x] = sdata[1]
    end 

    return 
end

function gpu_min_distance(Q::CuArray{Bool}, dist::CuArray{T}) where T
    nt = 256
    numblocks = ceil(Int, length(Q)/nt/2)
    smem = nt * sizeof(Float64) # shared memory per block
    out = CUDA.zeros(Float64, numblocks)
    CUDA.@sync begin
        @cuda threads = Int(nt) blocks = Int(numblocks) shmem = smem _gpu_min_distance!(Q, dist, out)
    end
    out = Array(out) # bring back to CPU
    return minimum(out) # do last reduction in the CPU (make custom function)
end

function BFM(G::Dict{Int64, Set{Int64}}, source, gr, U, fw::Function)

    n = length(G)
    p = Dict{Int, Int}() # best previous nodes
    # p = zeros(Int, n) # best previous nodes
    dist = fill(∞, n) # distance from best forward node
    dist[source] = 0
    # dist = [fw(gr[source], gr[i]) for i in 1:n]
    dist0 = deepcopy(dist)
    F = falses(n)
    for Gi in G[source]
        F[Gi] = true
    end
    it = 0

    # Initial active set is the neighbour set of the source(s)
    active = Set{Int}()
    union!(active, G[source])

    while it < n

        foo!(dist, p, dist0, G, active, gr, U, fw) 
        empty!(active)
        goo!(active, G, dist, dist0)
        if isempty(active)
            println("Converged in $it iterations")
            break
        end

        # foo1!(dist, p, dist0, G, F, gr, U, fw) 
        # fill!(F, false)
        # goo1!(F, G, dist, dist0)
        # length(active)
        # if sum(F) == 0
        #     # println("Converged in $it iterations")
        #     break
        # end

        copyto!(dist0, dist)
        it+=1
    end

    # converged = relaxation_BFM!(Q, F, G, dist, p, gr, U, fw)
        # if converged # or sum(Q) == 0
        #     break
        # end
        # update!(F, Q)
        # fill!(Q, false)
        # it+=1

    return Dijsktra(p, dist)
end

function foo1!(dist::Vector{T}, p, dist0, G, F, gr, U, fw) where T
    # Threads.@threads 
    for iactive in 1:length(F)
        @inbounds if F[iactive]
            di = dist0[iactive]
            for Gi in (G[iactive])
                tmp_dist = ifelse(
                    dist0[Gi] == ∞,
                    ∞,
                    dist0[Gi] + fw(gr[Gi], gr[iactive])*abs(U[Gi]-U[iactive])*0.5
                )
                if di > tmp_dist
                    di = tmp_dist
                    p[iactive] = Gi
                end
            end
            dist[iactive] = di 
        end
    end
end

function goo1!(active, G, dist, dist0)
    Threads.@threads for i in 1:length(G)
        @inbounds if dist[i] < dist0[i]
            for Gi in G[i]
                active[Gi] = true
            end
        end
    end
end

function foo!(dist::Vector{T}, p, dist0, G, active, gr, U, fw) where T
    
    Threads.@threads for iactive in collect(active)
        @inbounds di = dist0[iactive]
        @inbounds for Gi in (G[iactive])
            tmp_dist = ifelse(
                dist0[Gi] == ∞,
                ∞,
                dist0[Gi] + fw(gr[Gi], gr[iactive])/abs(U[Gi]+U[iactive])*0.5
            )
            if di > tmp_dist
                di = tmp_dist
                p[iactive] = Gi
            end
        end
        @inbounds dist[iactive] = di 
    end
end

function goo!(active, G, dist, dist0)
    lk = ReentrantLock()
    for i in 1:length(G)
         if dist[i] < dist0[i]
            lock(lk) do
                union!(active, G[i])
            end
        end
    end
end

function relaxation_BFM!(Q, F, G, dist, p, gr, U, fw::Function)
    converged = true
    Threads.@threads for i in eachindex(F)
        @inbounds if F[i]
            di = dist[i]
            for j in G[i]
                tmp_dist = di + fw(gr[i], gr[j])*abs(U[j]-U[i])*0.5
                if dist[j] > tmp_dist
                    dist[j] = tmp_dist
                    p[j] = i
                    Q[j] = true
                    converged = false
                end
            end
        end
    end
    return converged
end

function relaxation_BFM2!(Q, dist, G, p, gr, U, fw::Function)
    # Threads.@threads 
    for i in 1:length(Q)
        di = dist[i]
        Δdist = ∞
        index = -1
        for j in G[i]
            tempative_add = fw(gr[i], gr[j])*abs(U[j]-U[i])*0.5
            if tempative_add < Δdist
                Δdist = tempative_add
                index = j
            end
        end
        @show Δdist
        if Δdist < ∞
            dist[index] = di + Δdist
            p[index] = i
            Q[index] = true
        end
    end
end

function min_distance(Q, dist)
    d = Inf
    idx = -1
    for Qi in Q
        @inbounds di = dist[Qi]
        @inbounds if di < d
            d = di
            idx = Qi
        end
    end
    return idx
end

function BFM(G_d::CuGraph, source, gr_d::CuMesh, fw::Function)

    G, n1, n2 = G_d.K, G_d.n1, G_d.n2 # unpack graph
    x, y, z = gr_d.x, gr_d.y, gr_d.z # unpack mesh arrays
    U = gr_d.U # field 

    n = length(x) # number of nodes
    p = CUDA.fill(1, n) # best previous nodes
    dist = CUDA.fill(∞, n) # distance from best previous node
    dist[source] = 0 # distance at the source is 0
    converged = false
    F = CUDA.fill(false, n) # Frontier nodes
    F[source] = true
    Q = CUDA.fill(false, n) # Settled nodes

    for _ in 1:length(G)
        relaxation_BFM!(Q, F, G, dist, 
                        n1, n2, x, y, z, 
                        U, fw
        )
        update!(F, Q)
        converged = false
        CUDA.fill!(Q, false)
    end

    return CuDijsktra(p, dist)
end

function _gpu_relaxation_BFM!(Q, F, K, dist, p, n1, n2, x, y, z, U, fw)

    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if index < length(F)

        if F[index]

            xi, yi, zi = x[index], y[index], z[index]
            ileft, iright = n1[index], n2[index]

            # TODO cache out xj yj zj ?
            for j in ileft:iright
                if Q[K[j]]
                    tmp_dist = dist[index] + fw(xi, yi, zi, x[j], y[j], z[j])*abs(U[j]-U[index])*0.5
                    sync_threads()
                    if dist[j] > tmp_dist
                        dist[j] = tmp_dist
                        p[j] = index
                        Q[j] = true
                    end
                    sync_threads()
                end
            end

        end
    end

    return nothing
end

function relaxation_BFM!(Q::CuArray{Bool}, F::CuArray{Bool}, G::CuArray, dist::CuArray, 
                        n1::CuArray, n2::CuArray, x::CuArray, y::CuArray, z::CuArray, 
                        U::CuArray, fw::Function
                        )
    
    nt = 256
    numblocks = ceil(Int, length(F)/nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_relaxation_BFM!(Q, F, G, dist, p, n1, n2, x, y, z, U, fw)
    end
end

function update!(F, Q)
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] = false
        @inbounds if (Q[i] == true)
            Q[i] = false
            F[i] = true
        end
    end
end

function _gpu_update!(F, Q)
    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if index < length(F)
        F[index] = false
        if Q[index] == true
            Q[index] = false
            F[index] = true
        end
    end
    return 
end

function update!(F::CuArray{Bool}, Q::CuArray{Bool}) 
    nt = 256
    numblocks = ceil(Int, length(F)/nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gpu_update!(F, Q)
    end
end

function move2device(K, U, gr)

    # move velocity field to device
    U_d = CuArray(Float32.(U))

    # move spatial coordinates arrays to device (TODO specialize for lazy grids)
    x_d = CuArray([gr[i].x for i in 1:Π(gr.nnods)])
    y_d = CuArray([gr[i].y for i in 1:Π(gr.nnods)])
    z_d = CuArray([gr[i].z for i in 1:Π(gr.nnods)])

    # Setup Graph arrays
    K_d = CuVector(K.rowval) # sparse nodal connectivity
    # indices of K corresponding to nodes connected to the i-th node
    nz1 = Vector{Int64}(undef, K.n)
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
    mesh_d = CuMesh(x_d, y_d, z_d, U_d)

    return graph_d, mesh_d

end