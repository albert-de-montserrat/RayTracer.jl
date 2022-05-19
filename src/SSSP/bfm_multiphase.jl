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
    # Uboundary = ray_direction == :above ? interpolant(r_boundary + buffer_zone) : interpolant(r_boundary - buffer_zone)
    Uboundary = if ray_direction == :above
        interpolant(r_boundary - buffer_zone)
    else
        interpolant(r_boundary + buffer_zone)
    end
    U[boundary_nodes] .= Uboundary
    return U
end

function bfm_multiphase(
    Gsp::SparseMatrixCSC,
    source::Int,
    gr,
    U::Vector{T},
    partition::GridPartition,
    interpolant,
) where {T}

    # unpack partition
    ID = partition.id
    rboundaries = partition.rboundaries
    nlayers = partition.nlayers
    nboundaries = partition.nboundaries
    layers = partition.layers
    boundaries = partition.boundaries
    iterator = partition.iterator

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

    # # update velocity at boundary
    # boundary_velocity!(U, interpolant, rboundaries[1], boundary_nodes[1], :below) 

    # unpack coordinates
    x, z = gr.x, gr.z

    # number of nodes in the graph
    n = length(x)

    # allocate dictionary containing best previous nodes
    p = Vector{Int}(undef, n)

    # priority queue containing NOT settled nodes
    Q = falses(n)
    # # 1st frontier nodes are nodes adjacent to the source
    # @inbounds for Gi in @views Gsp.rowval[nzrange(Gsp, source)]
    #     # ID[Gi] ∉ current_level && continue
    #     Q[Gi] = true
    # end

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # forward loop
    for i in 1:3 #length(iterator)
        current_level = iterator[i]
        current_boundary = if length(current_level) == 2
            tuple(current_level[2])
        else
            tuple(current_level[2], current_level[3])
        end

        # update velocity at current boundaries
        for j in current_boundary
            boundary_velocity!(
                U, interpolant, boundary_dict[j], boundary_nodes[j], ray_direction[i]
            )
        end

        if i > 1
            # update source
            # source = find_new_source_min(dist, ID, current_boundary[1])
            # reset distances, except at the boundary
            # if length(current_boundary) == 1 
            #     # @show current_boundary, i
            #     dist[setdiff(1:gr.nnods, boundary_nodes[current_boundary[1]])] .= typemax(T)

            # else
            #     # @show current_boundary, i
            #     # ikeep = ifelse(
            #     #     i > nlayers,
            #     #     boundary_nodes[current_boundary[2]],
            #     #     boundary_nodes[current_boundary[1]],
            #     # )
            #     ikeep = boundary_nodes[current_boundary[1]]
            #     dist[setdiff(1:gr.nnods,ikeep)] .= typemax(T)
            # end
            # dist[setdiff(1:gr.nnods, boundary_nodes[current_boundary[1]])] .= typemax(T)

        end

        @inbounds for Gi in @views Gsp.rowval[nzrange(Gsp, source)]
            ID[Gi] ∉ current_level && continue
            Q[Gi] = true
        end

        # covergence: if the queue is empty we are done
        it = 0
        while sum(Q) != 0
            # update iteration counter
            it += 1

            # relax edges (parallel process)
            _relax_bfm!(dist, p, dist0, Gsp, Q, x, z, U)

            # pop queue 
            fillfalse!(Q)

            # update nodal queue (parallel process)
            _update_Q!(Q, Gsp, dist, dist0, ID, current_level)

            # update old distance vector (TODO parallel version)
            copyto!(dist0, dist)
        end

        println("Level $i converged in $it iterations")
        # pop queue 
        fillfalse!(Q)
    end

    # println("Total $it iterations")

    return BellmanFordMoore(p, dist)
end

function find_new_source_min(dist::Vector{T}, ID, boundary) where {T}
    di = typemax(T)
    idx = -1
    for i in eachindex(dist)
        @inbounds if ID[i] == boundary
            if dist[i] < di
                di = dist[i]
                idx = i
            end
        end
    end
    return idx
end

function find_new_source_max(dist::Vector{T}, ID, boundary) where {T}
    di = typemax(T)
    idx = -1
    for i in eachindex(dist)
        @inbounds if ID[i] == boundary
            if dist[i] < di
                di = dist[i]
                idx = i
            end
        end
    end
    return idx
end

function _update_Q!(
    Q::BitVector, G::SparseMatrixCSC, dist, dist0, ID, current_level::NTuple
)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in eachindex(Q)
        @inbounds if dist[i] < dist0[i]
            for Gi in @views G.rowval[nzrange(G, i)]
                ID[Gi] ∉ current_level && continue
                Q[Gi] = true
            end
        end
    end
end
