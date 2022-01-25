function element_neighbours(gr)
    (; e2n, nnods, neighbour) = gr
    nn = sum([length(n) for el in els])
    nel = size(els,2)

    # Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    @inbounds for i in axes(els,1), j in axes(els,2)
        node = els[i,j]
        push!(I, node)
        push!(J, j)
        push!(V, true)
    end
    incidence_matrix = sparse(J, I, V)

    # Find neighbouring elements
    neighbour = [Int64[] for _ in 1:nel]
    nnod = I[end]
    @inbounds for node in 1:nnod
        # Get elements sharing node
        r = nzrange(incidence_matrix, node)
        el_neighbour = incidence_matrix.rowval[r]
        # Add neighbouring elements to neighbours map
        for iel1 in el_neighbour
            current_neighbours = neighbour[iel1]
            for iel2 in el_neighbour
                # check for non-self neighbour and avoid repetitions
                if (iel1!=iel2) && (iel2 ∉ current_neighbours)
                    push!(neighbour[iel1],iel2)
                end
            end
        end
    end

    return neighbour
end

function primary_grid_chunk(nθ, nr)
    
    θrange = (0, 30)
    L = 100
    r_out = R
    r_in = R-L
    nn = nr*nθ # total number of nodes
    nels = (nr-1)*nθ
    r = fill(0.0, nn)
    θ = fill(0.0, nn)
    dθ = deg2rad(abs(θrange[2]-θrange[1]))/nθ
    # -- Nodal positions
    @inbounds for ii in 1:nθ
        idx = @. (1:nr) + nr*(ii-1)
        r[idx] .= LinRange(r_in, r_out, nr)
        θ[idx] .= dθ*(ii-1)
    end
    
    # -- Quadrilateral elements
    id_el = Matrix{Int64}(undef, nels, 4)
    @inbounds for ii in 1:nθ
        idx  = @. (1:nr-1) + (nr-1)*(ii-1)
        idx1 = @. (1:nr-1) + nr*(ii-1)
        idx2 = @. (1:nr-1) + nr*(ii)
        id_el[idx,:] .= [idx1 idx2 idx2.+1 idx1.+1]
    end

    # -- connectivity and element type
    element_type = Dict{Int, Symbol}()
    connectivity = Dict{Int, Vector{Int64}}()
    @inbounds for ii in 1:nels # rectangular elements
        connectivity[ii] = id_el[ii,:]
        element_type[ii] = :Quad
    end

    neighbours = element_neighbours(connectivity)
    nel = length(connectivity)
    x, z = @cartesian(θ, r)

    gr = Grid2D(x,
        z,
        θ,
        r,
        connectivity,
        nθ,
        nr,
        nel,
        length(x),
        neighbours,
        element_type
    )

    return gr
end

function edge_connectivity_chunk(gr)
    e2n, neighbours = gr.e2n, gr.neighbours
    nel = gr.nel

    
    local_map = [ 1 2 3 4
                  2 3 4 1 ]

    global_idx = 0
    el2edge = fill(0, 4, nel)
    edge2node = Vector{Int64}[]
    edge_neighbour = Matrix{Int64}(undef, 2, 4)

    edge2el = Dict{Int, Set{Int}}()
    @inbounds for iel in 1:nel
        element = e2n[iel]
    
        edge = e2n[iel][local_map]
        sort!(edge, dims=1)
        
        # neighbours of local element
        el_neighbours = neighbours[iel]
        edge_neighbours = [e2n[i][local_map] for i in el_neighbours]

        # check edge by edge
        nedge = length(element)
        for iedge in 1:nedge
            if el2edge[iedge, iel] == 0
                global_idx += 1
                el2edge[iedge, iel] = global_idx
                push!(edge2node, view(edge,:, iedge))

                if !haskey(edge2el, global_idx)
                    edge2el[global_idx] = Set{Int}()
                end
                push!(edge2el[global_idx], iel)

                # edges the neighbours
                for (c, ieln) in enumerate(el_neighbours)
                    edge_neighbour = edge_neighbours[c]
                    sort!(edge_neighbour, dims=1)

                    # check wether local edge is in neighbouring element
                    for i in 1:nedge
                        if issubset(edge[:, iedge], edge_neighbour)
                            el2edge[i, ieln] = global_idx
                            push!(edge2el[global_idx], ieln)
                            break
                        end
                    end

                end

            end
        end

    end

    # Make dictionaries
    edge_nodes = Dict{Int, NTuple{2, Int}}()
    @inbounds for i in 1:length(edge2node)
        edge_nodes[i] = (edge2node[i][1], edge2node[i][2])
    end
    element_edge = Dict{Int, Vector{Int}}()
    @inbounds for i in axes(el2edge,2)
        idx = findlast(view(el2edge, :, i) .> 0)
        if idx == 4
            element_edge[i] = el2edge[:, i]

        else
            element_edge[i] = el2edge[1:idx, i]
        end
    end

    return element_edge, edge_nodes, edge2el

end


####
function bfmtest(G::SparseMatrixCSC{Bool, M}, source::Int, gr, U::Matrix{T}) where {M,T}

    # unpack coordinates
    (; e2n, x, z, r) = gr

    # number of nodes in the graph
    n = G.n

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
    init_Q!(Q, G, e2n, source)

    # initialise all distances as infinity and zero at the source node 
    dist = fill(typemax(T), n)
    dist[source] = zero(T)
    dist0 = deepcopy(dist)

    # main lopp
    it = 1
    # covergence: if the queue is empty we are done
    @inbounds while sum(Q) != 0
        
        # relax edges (parallel process)
        # relax!(dist, p, dist0, G, Q, tmp, e2n, x, z, r, U)
        relax!(dist, p, dist0, G, Q, e2n, x, z, r, U)
            
        # pop queue (serial-but-fast process)
        fillfalse!(Q)

        # update nodal queue (parallel process)
        update_Q!(Q, G, dist, dist0, e2n)

        # update old distance vector (TODO parallel version)
        copyto!(dist0, dist)

        # update iteration counter
        it+=1
    end

    println("Converged in $it iterations")

    return BellmanFordMoore(p, dist)
end

function init_Q!(Q::BitVector, G::SparseMatrixCSC{Bool, T}, e2n::Dict, source::Integer) where T
    for element in @views G.rowval[nzrange(G, source)]
        for i in e2n[element]
            Q[i] = true
        end
    end
end

function update_Q!(Q::BitVector, G::SparseMatrixCSC, dist, dist0, e2n)
    # update queue: if new distance is smaller 
    # than the previous one, add to adjecent 
    # to the queue nodes
    Threads.@threads for i in findall(dist.<Inf)
        # Threads.@threads for i in eachindex(Q)
        if dist[i] < dist0[i]
            for element in @views G.rowval[nzrange(G, i)]
                for i in e2n[element]
                    Q[i] = true
                end
            end
        end
    end
end

@inline function relax!(dist::Vector{T}, p::Vector, dist0, G::SparseMatrixCSC, Q::BitVector, e2n, x, z, r, U::Matrix) where T
    # iterate over queue. Unfortunately @threads can't iterate 
    # over a Set, so we need to collect() it. This yields an 
    # allocation, but it's worth it in this case as it saves 
    # a decent number of empty iterations and removes a layer
    # of branching
    Threads.@threads for i in findall(Q)
        _relax!(p, dist, dist0, G, i, e2n, x, z, r, U)
    end
end

function adjacents!(tmp, G, idx, e2n)
    N = 0
    indices = @views G.rowval[nzrange(G, idx)]
    @inbounds for j in indices
        for Gi in e2n[j]
            if Gi ∉ view(tmp, 1:N+1)
                N+=1
                tmp[N+1] = Gi
            end
        end
    end
    tmp[1] = N
end

@inbounds function _relax!(p::Vector, dist, dist0, G::SparseMatrixCSC, i, e2n, x, z, r, U::Matrix{T}) where T
   
    # cache coordinates, velocity and distance of frontier node
    di = dist0[i]
    xi, zi, ri = x[i], z[i], r[i]
    Ui = (U[i, 1], U[i, 2])
    
    # iterate over adjacent nodes to find the the one with 
    # the mininum distance to current node
    for j in @views G.rowval[nzrange(G, i)]
        for Gi in e2n[j] 
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

            # update distance and predecessor index 
            # if it's smaller than the temptative distance
            if di > δ
                di = δ
                p[i] = Gi
            end
        end
    end

    # update distance
    dist[i] = di 
end

# function list_by_distance(gr, nθ, nr)
#     dθ, dr = 2*2*π/nθ, 2*R/nr

#     tmp = zeros(Int, 10_000)
    
#     for inod in 1:gr.nnods
#         N = 0
#         node = gr[inod]
#         θ1 = node.x - dθ < 0 ? node.x - dθ + 2π : node.x - dθ
#         θ2 = node.x + dθ > 2π ? node.x + dθ - 2π : node.x + dθ
#         r1 = node.z - dr
#         r2 = node.z + dr

#         for i in 1:gr.nnods
#             if (gr[i] != node) && (gr[i].x ≥ θ1) && (gr[i].x ≤ θ2) && (r2 ≥ gr[i].z ≥ r1)
#                 N += 1
#                 tmp[N] = i
#             end
#         end
#     end
# end


function guess_dimension(gr::Grid2D)
    e2n = gr.e2n
    nels = gr.nel
   
    # count = zeros(Int64, gr.nnods)
    count = 0
    @inbounds for i in 1:nels
        element_nodes = e2n[i]
        # count[element_nodes] .+= length(element_nodes)  
        count += length(element_nodes)*1
    end
    
    return count
end

function test(gr)
    guess = guess_dimension(gr)

    # Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    sizehint!(I, guess)
    sizehint!(J, guess)
    sizehint!(V, guess)
    
    @inbounds for iel in 1:gr.nel
        element = gr.e2n[iel]
         
        for j in element
            for k in element
                if k!=j
                    push!(I, k)
                    push!(J, j)
                    push!(V, true)
                end
            end
        end
    end

    # A = sparse(I, J, V)
end