# wave direction 
function directions(nlayers)
    nmax = 2nlayers - 1
    ray_direction = Dict{Int,NTuple{2,Symbol}}()
    ray_direction[1] = ray_direction[nmax] = (:above, :above)
    for i in 2:(nlayers - 1)
        # For convinience, the put first the boundary where the SSSP is going to be restarted
        ray_direction[i] = ray_direction[nmax - i + 1] = (:below, :above)
        # ray_direction[i] = ray_direction[nmax-i+1] = (:above, :below)
        # ray_direction[nmax-i+1] = (:below, :above)
    end
    ray_direction[nlayers] = (:below, :below)
    return ray_direction
end

function boundary_velocity!(
    U, interpolant::Interpolations.Extrapolation, r_boundary, boundary_nodes, ray_direction
)
    buffer_zone = 1 # in km
    Uboundary = if ray_direction == :above
        interpolant(r_boundary + buffer_zone)
    else
        interpolant(r_boundary - buffer_zone)
    end
    # Uboundary = ray_direction == :above ? interpolant(r_boundary - buffer_zone) : interpolant(r_boundary + buffer_zone)
    U[boundary_nodes] .= Uboundary
    return U
end

function bfm_ms(G::SparseMatrixCSC{Bool,M}, halo, source::Int, gr, U::Vector{T}) where {M,T}
    partition = partition_grid(gr)
    # unpack partition
    (; id, rboundaries, nlayers, nboundaries, layers, boundaries, iterator) = partition
    boundary_dict = Dict(
        a => b for (a, b) in zip(partition.boundaries, partition.rboundaries)
    )
    # find ids of boundary nodes
    boundary_nodes = Dict(
        a => findall(round.(gr.r, digits=2) .== b) for
        (a, b) in zip(partition.boundaries, partition.rboundaries)
    )
    # wether to take the velocity just above or below the discontinuity
    ray_direction = directions(nlayers)

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

    # allocate dictionary containing best previous nodes
    # p = Dict{M, M}()
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
    for i in [1, length(iterator)]
        current_level = iterator[i]
        current_boundary = if length(current_level) == 2
            tuple(current_level[2])
        else
            tuple(current_level[2], current_level[3])
        end

        # # update velocity at current boundaries
        # for j in current_boundary
        #     boundary_velocity!(U, interpolant, boundary_dict[j], boundary_nodes[j], ray_direction[i]) 
        # end

        if i > 1
            # update source
            source = find_new_source_min(dist, id, current_boundary[1])
            # reset distances, except at the boundary
            if length(current_boundary) == 1
                # @show current_boundary, i
                dist[setdiff(1:(gr.nnods), boundary_nodes[current_boundary[1]])] .= typemax(
                    T
                )

            else
                # @show current_boundary, i
                # ikeep = ifelse(
                #     i > nlayers,
                #     boundary_nodes[current_boundary[2]],
                #     boundary_nodes[current_boundary[1]],
                # )
                ikeep = boundary_nodes[current_boundary[1]]
                dist[setdiff(1:(gr.nnods), ikeep)] .= typemax(T)
            end
            dist[setdiff(1:(gr.nnods), boundary_nodes[current_boundary[1]])] .= typemax(T)
            init_Q!(Q, G, e2n, source)
        end

        # for element in sp_column(G, source)
        #     for i in e2n[element]
        #         id[i] ∉ current_level && continue
        #         Q[i] = true
        #     end
        # end

        # covergence: if the queue is empty we are done
        @inbounds while sum(Q) != 0

            # relax edges (parallel process)
            relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)

            # update discontinuous elements
            update_halo!(p, dist, dist0, halo)

            # pop queue (serial-but-fast process)
            fillfalse!(Q)

            # update nodal queue (parallel process)
            _update_Q!(Q, G, dist, dist0, e2n, id, current_level)

            # update old distance vector (TODO parallel version)
            copyto!(dist0, dist)

            # update iteration counter
            it += 1
        end
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

sp_column(A::SparseMatrixCSC, I::T) where {T<:Integer} = @views A.rowval[nzrange(A, I)]

function init_Q!(
    Q::BitVector, G::SparseMatrixCSC{Bool,T}, e2n::Dict, source::Integer
) where {T}
    for element in sp_column(G, source)
        for i in e2n[element]
            Q[i] = true
        end
    end
end

function _update_Q!(
    Q::BitVector, G::SparseMatrixCSC, dist, dist0, e2n, ID, current_level::NTuple
)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        if (dist[i] < Inf) && (dist[i] < dist0[i])
            for element in sp_column(G, i)
                for i in e2n[element]
                    ID[i] ∉ current_level && continue
                    Q[i] = true
                end
            end
        end
    end
end

function update_Q!(Q::BitVector, G::SparseMatrixCSC, dist, dist0, e2n)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in findall(dist .< Inf)
        if dist[i] < dist0[i]
            for element in sp_column(G, i)
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
    U::Vector,
) where {T}
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        _relax!(p, dist, dist0, G, i, e2n, x, z, r, U)
    end
end

@inbounds function _relax!(
    p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, U::Vector{T}
) where {T}

    # read coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi = x[i], z[i]
    Ui = U[i]

    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in sp_column(G, i), Gi in e2n[j]
        dGi = dist0[Gi]
        # i is the index of the ray-tail, Gi index of ray-head
        # branch-free arithmetic to check whether ray is coming from above or below
        # idx = 1 if ray is going downards, = 2 if going upwards
        # head_idx = (ri > r[Gi]) + 1
        # tail_idx = (head_idx==1) + 1
        # temptative distance (ignore if it's ∞)
        δ = ifelse(
            dGi == typemax,
            typemax(T),
            muladd(2, distance(xi, zi, x[Gi], z[Gi]) / (Ui + U[Gi]), dGi),
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
