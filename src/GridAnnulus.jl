abstract type Cartesian end
abstract type Polar end

struct Point2D{T}
    x::Float64
    z::Float64
end

struct Grid2D{A,B,C,D,E}
    x::A
    z::A
    θ::A
    r::A
    e2n::B
    nθ::C
    nr::C
    nel::C
    nnods::C
    neighbours::D
    element_type::E
end

Base.length(gr::Grid2D) = gr.nnods

Base.getindex(gr::Grid2D, I) = Point2D{Polar}(gr.θ[I], gr.r[I])

macro cartesian(x::Union{Vector, Symbol, Expr}, z::Union{Vector, Symbol, Expr}) 
    esc(:( @.($z*sin($x)), @.($z*cos($x)) )) 
end

function cartesian2polar(x::T, z::T) where T<:Real 
    θ = atan(x,z)
    if θ < 0
        θ += 2π
    end 
    return (θ, sqrt(x^2+z^2))
end

function cartesian2polar(x::Vector, z::Vector) where T<:Real 
    θ = @. atan(x,z)
    for i in eachindex(θ)
        if θ[i] < 0
            θ[i] +=2π
        end 
    end
    return (θ, @.(sqrt(x^2+z^2)))
end

polar2cartesian(x::T, z::T) where T<:Real = (z*sin(x), z*cos(x))

polar2cartesian(x::Vector, z::Vector) = (@. z*sin(x)), (@. z*cos(x))

function primary_grid(nθ, nr, r_out)
    
    nn = nr*nθ # total number of nodes
    nels = (nr-1)*nθ
    r = fill(0.0, nn+1)
    θ = fill(0.0, nn+1)
    dθ = 2*π/nθ
    dr = r_out/nr 
    r_in = r_out - dr*(nr-1)
    # r_in = 2891.5f0
    # -- Nodal positions
    @inbounds for ii in 1:nθ
        idx = @. (1:nr) + nr*(ii-1)
        r[idx] .= LinRange(r_in, r_out, nr)
        θ[idx] .= dθ*(ii-1)
    end
    # center of the core
    r[end], θ[end] = 0.0, 0.0

    # -- Quadrilateral elements
    id_el = Matrix{Int64}(undef, nels, 4)
    @inbounds for ii in 1:nθ
        if ii < nθ
            idx  = @. (1:nr-1) + (nr-1)*(ii-1)
            idx1 = @. (1:nr-1) + nr*(ii-1)
            idx2 = @. (1:nr-1) + nr*(ii)
            id_el[idx,:] .= [idx1 idx2 idx2.+1 idx1.+1]
        else
            idx  = @. (1:nr-1) + (nr-1)*(ii-1)
            idx1 = @. (1:nr-1) + nr*(ii-1)
            idx2 = @. 1:nr-1
            id_el[idx,:] .= [idx1 idx2 idx2.+1 idx1.+1]
        end
    end

    # -- Triangular elements
    id_triangle = Matrix{Int64}(undef, nθ, 3)
    @inbounds for ii in 1:nθ
        idx = 1 + nr*(ii-1)
        id_triangle[ii, 1] = nn+1
        id_triangle[ii, 2] = idx
        id_triangle[ii, 3] = idx + nr
    end
    id_triangle[end] = 1

    # -- connectivity and element type
    element_type = Dict{Int, Symbol}()
    connectivity = Dict{Int, Vector{Int64}}()
    @inbounds for ii in 1:nels # rectangular elements
        connectivity[ii] = id_el[ii,:]
        element_type[ii] = :Quad
    end
    @inbounds for (i, ii) in enumerate(1+nels:nθ+nels) # triangular elements
        connectivity[ii] = id_triangle[i,:]
        element_type[ii] = :Tri
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

function add_discontinuities(gr, spacing, dθ, dr)
    # add velocity boundaries
    layers = velocity_layers(gr, spacing)
    global_idx = nnods0 = length(gr.x)
    neighbours = gr.neighbours
    for l in layers
        # cartesian coordinates of the boundary
        # θl, rl = l[1], l[2]
        θl, rl = cartesian2polar(l[1], l[2])
        for (θi, ri) in zip(θl, rl)
            # new node global index
            global_idx += 1
            # find element where new node belongs
            θreps = Int(fld(θi, dθ))
            rreps = Int(cld(ri, dr))
            iel = (nr-1)*θreps + rreps - 1
            # add to connectivity matrix
            push!(gr.e2n[iel], global_idx)
            # add node to 2 levels of adjacency
            for i1 in neighbours[iel]
                if global_idx ∉ gr.e2n[i1]
                    push!(gr.e2n[i1], global_idx)
                end
                # for i2 in neighbours[i1]
                #     if global_idx ∉ gr.e2n[i2]
                #         push!(gr.e2n[i2], global_idx)
                #     end
                # end
            end
            # add  to elements on both sides
            if θi == 0
                θreps = Int(fld(2π, dθ))
                iel_n = (nr-1)*θreps + rreps - 1
                push!(gr.e2n[iel_n], global_idx)
                 # add node to 2 levels of adjacency
                # for i1 in neighbours[iel_n]
                #     if global_idx ∉ gr.e2n[i1]
                #         push!(gr.e2n[i1], global_idx)
                #     end
                #     # for i2 in neighbours[i1]
                #     #     if global_idx ∉ gr.e2n[i2]
                #     #         push!(gr.e2n[i2], global_idx)
                #     #     end
                #     # end
                # end
            end
        end
    end

    # concatenate nodes of velocity boundaries
    lθ = [cartesian2polar(l[1], l[2]) for l in layers]
    θboundary = reduce(vcat, l[1] for l in lθ)
    rboundary = reduce(vcat, l[2] for l in lθ)
    # convert to polar coordinates
    xboundary, zboundary = polar2cartesian(θboundary, rboundary)

    gr = Grid2D(
        vcat(gr.x, xboundary),
        vcat(gr.z, zboundary),
        vcat(gr.θ, θboundary),
        vcat(gr.r, rboundary),
        gr.e2n,
        gr.nθ,
        gr.nr,
        gr.nel,
        gr.nnods + length(xboundary),
        gr.neighbours,
        gr.element_type
    )

    G = nodal_incidence(gr, star_levels = 1)

    # constrain2layers!(G, gr)

    return G, gr, nnods0

end

function expand_secondary_nodes!(G::Dict{T, Set{T}}, gr, nnods0) where T
    Q = Set{T}()
    # degrees = nodal_degree(G)
    # max_degree = maximum(degrees[(nnods0+1):end])
    # Q = Vector{T}(undef, max_degree*100)
    # expand adjency of boundary nodes
    for _ in 1:1
        G0 = deepcopy(G)
        for i in (nnods0+1):gr.nnods
            # c = 0 # counter
            for idx in G0[i]
                # c+=1
                # nodes2add =  
                # Q[1] = 
                # union!(G[i], G02[idx])
                union!(Q, G0[idx])
                # for idx2 in G0[idx]
                #     union!(Q, G0[idx2])
                # end
            end
            union!(G[i], Q)
            empty!(Q)
        end
        # constrain2layers!(G, gr)
    end
end

function init_annulus(
    nθ::Int64, nr::Int64; spacing = 20, r_out = 6371.0, r_in = 6371-5153.5,
    star_levels = 0
    )

    # grid containing only primary nodes
    gr = primary_grid(nθ, nr, r_out)
    
    gr = secondary_nodes(gr, spacing = spacing)

    nnods0 = length(gr.x)
    dθ, dr = 2*π/nθ, r_out/nr 
    G, gr, nnods0 = add_discontinuities(gr, spacing, dθ, dr)
    # expand_secondary_nodes!(G, gr, nnods0)
    cleanse_graph!(G)

    # # reorder nodes
    # degrees = nodal_degree(G)
    # prm = symrcm(G, degrees)
    # reorder!(gr, prm)
    # G = nodal_incidence(gr, star_levels = 1)

    constrain2layers!(G, gr)

    return gr, G
end

# function init_annulus(
#     nθ::Int64, nr::Int64; spacing = 20, r_out = 6371f0, r_in = 6371f0-5153.5f0,
#     star_levels = 0
#     )
     
#     # r_out = r_out
#     # # r_in = r_in
#     # nn = nr*nθ # total number of nodes
#     # nels = (nr-1)*nθ
#     # r = fill(0.0, nn+1)
#     # θ = fill(0.0, nn+1)
#     # dθ = 2*π/nθ
#     # dr = r_out/nr 
#     # r_in = r_out - dr*(nr-1)

#     # # -- Nodal positions
#     # @inbounds for ii in 1:nθ
#     #     idx = @. (1:nr) + nr*(ii-1)
#     #     r[idx] .= LinRange(r_in, r_out, nr)
#     #     θ[idx] .= dθ*(ii-1)
#     # end
#     # # center of the core
#     # r[end], θ[end] = 0.0, 0.0

#     # # -- Quadrilateral elements
#     # id_el = Matrix{Int64}(undef, nels, 4)
#     # @inbounds for ii in 1:nθ
#     #     if ii < nθ
#     #         idx  = @. (1:nr-1) + (nr-1)*(ii-1)
#     #         idx1 = @. (1:nr-1) + nr*(ii-1)
#     #         idx2 = @. (1:nr-1) + nr*(ii)
#     #         id_el[idx,:] .= [idx1 idx2 idx2.+1 idx1.+1]
#     #     else
#     #         idx  = @. (1:nr-1) + (nr-1)*(ii-1)
#     #         idx1 = @. (1:nr-1) + nr*(ii-1)
#     #         idx2 = @. 1:nr-1
#     #         id_el[idx,:] .= [idx1 idx2 idx2.+1 idx1.+1]
#     #     end
#     # end

#     # # -- Triangular elements
#     # id_triangle = Matrix{Int64}(undef, nθ, 3)
#     # @inbounds for ii in 1:nθ
#     #     idx = 1 + nr*(ii-1)
#     #     id_triangle[ii, 1] = nn+1
#     #     id_triangle[ii, 2] = idx
#     #     id_triangle[ii, 3] = idx + nr
#     # end
#     # id_triangle[end] = 1

#     # # -- connectivity and element type
#     # element_type = Dict{Int, Symbol}()
#     # connectivity = Dict{Int, Vector{Int64}}()
#     # @inbounds for ii in 1:nels # rectangular elements
#     #     connectivity[ii] = id_el[ii,:]
#     #     element_type[ii] = :Quad
#     # end
#     # @inbounds for (i, ii) in enumerate(1+nels:nθ+nels) # triangular elements
#     #     connectivity[ii] = id_triangle[i,:]
#     #     element_type[ii] = :Tri
#     # end

#     # neighbours = element_neighbours(connectivity)
#     # nel = length(connectivity)
#     # x, z = @cartesian(θ, r)

#     # gr = Grid2D(x,
#     #     z,
#     #     θ,
#     #     r,
#     #     connectivity,
#     #     nθ,
#     #     nr,
#     #     nel,
#     #     length(x),
#     #     neighbours,
#     #     element_type
#     # )
    
#     # # grid containing only primary nodes
#     # gr = primary_grid(nθ, nr, r_out)
    
#     # G0 = nodal_incidence(gr, star_levels = 1)

#     # # degrees = nodal_degree(G0)
#     # # prm = symrcm(G0, degrees)
#     # # reorder!( gr, prm)

#     # # G0 = nodal_incidence(gr, star_levels = 1)

#     # gr = secondary_nodes(gr, spacing = spacing)

#     # G02 = deepcopy(G)

#     # degrees = nodal_degree(G)
#     # prm = symrcm(G, degrees)
#     # reorder!( gr, prm)

#     # G = nodal_incidence(gr, star_levels = 1)
    
#     # # add velocity boundaries
#     # layers = velocity_layers(gr, spacing)
#     # global_idx = nnods0 = length(gr.x)
#     # for l in layers
#     #     # cartesian coordinates of the boundary
#     #     # θl, rl = l[1], l[2]
#     #     θl, rl = cartesian2polar(l[1], l[2])
#     #     for (θi, ri) in zip(θl, rl)
#     #         # new node global index
#     #         global_idx += 1
#     #         # find element where new node belongs
#     #         θreps = Int(fld(θi, dθ))
#     #         rreps = Int(cld(ri, dr))
#     #         iel = (nr-1)*θreps + rreps - 1
#     #         # add to connectivity matrix
#     #         push!(gr.e2n[iel], global_idx)
#     #         # add  to elements on both sides
#     #         if θi == 0
#     #             θreps = Int(fld(2π, dθ))
#     #             push!(gr.e2n[(nr-1)*θreps + rreps - 1], global_idx)
#     #         end
#     #     end
#     # end

#     # # concatenate nodes of velocity boundaries
#     # lθ = [cartesian2polar(l[1], l[2]) for l in layers]
#     # θboundary = reduce(vcat, l[1] for l in lθ)
#     # rboundary = reduce(vcat, l[2] for l in lθ)
#     # # convert to polar coordinates
#     # xboundary, zboundary = polar2cartesian(θboundary, rboundary)

#     # gr = Grid2D(
#     #     vcat(gr.x, xboundary),
#     #     vcat(gr.z, zboundary),
#     #     vcat(gr.θ, θboundary),
#     #     vcat(gr.r, rboundary),
#     #     gr.e2n,
#     #     nθ,
#     #     nr,
#     #     nel,
#     #     gr.nnods + length(xboundary),
#     #     gr.neighbours,
#     #     gr.element_type
#     # )

#     # G = nodal_incidence(gr, star_levels = 1)

#     # constrain2layers!(G, gr)

#     # grid containing only primary nodes
#     gr = primary_grid(nθ, nr, r_out)
    
#     # degrees = nodal_degree(G0)
#     # prm = symrcm(G0, degrees)
#     # reorder!( gr, prm)
#     # G0 = nodal_incidence(gr, star_levels = 1)

#     gr = secondary_nodes(gr, spacing = spacing)

#     nnods0 = length(gr.x)
#     dθ, dr = 2*π/nθ, r_out/nr 
#     G, gr, nnods0 = add_discontinuities(gr, spacing, dθ, dr)
#     # expand_secondary_nodes!(G, gr, nnods0)

#     # expand adjency of boundary nodes
#     # for _ in 1:2
#     #     G02 = deepcopy(G)
#     #     for i in (nnods0+1):gr.nnods
#     #         for idx in G02[i]
#     #             union!(G[i], G02[idx])
#     #         end
#     #     end
#     #     constrain2layers!(G, gr)
#     # end

#     # if star_levels > 0 
#     #     add_star_levels!(G, G0, star_levels)
#     # end

#     cleanse_graph!(G)

#     # # reorder nodes
#     # degrees = nodal_degree(G)
#     # prm = symrcm(G, degrees)
#     # reorder!(gr, prm)
#     # G = nodal_incidence(gr, star_levels = 1)

#     constrain2layers!(G, gr)

#     # make sparse array
#     # Gsp = graph2sparse(G)

#     return gr, G
# end

function add_star_levels!(G, G0, star_levels)
    # G0 = deepcopy(G)
    for _ in 1:star_levels
        G02 = deepcopy(G)
        for (i, G0i) in G0
            for j in G0i
                union!(G[i], G02[j])
            end
            delete!(G[i],i)
        end
    end
end

function velocity_layers(gr, spacing; npoints = 180)
    r = R.-(20f0, 35f0, 210f0, 410f0, 660f0, 2740f0, 2891.5f0)

    npoints = [Int(ri*2*π ÷ spacing) for ri in r]
    npoints[1:2].*=2
    layers = [circle(np, ri, pop_end = true) for (ri, np) in zip(r, npoints)]

    # # layers = [circle(npoints, r, pop_end = false) for r in r]
    # idx = gr.r.== R
    # n = sum(idx)
    # θl = gr.θ[idx]
    # layers = [(θl, fill(r[1], n)) for r in r]

    return layers
end

function constrain2layers!(G, gr)
    rlayer = (R, R-20, R-35, R-210, R-410, R-660, R-2740, R-2891.5)
    r = gr.r

    @inbounds for (i, Gi) in G
        ri = round(r[i], digits=1)
        # ri = r[i]
        
        if ri ∉ rlayer
            upper_limit, lower_limit = find_layer(ri, rlayer)

            for idx in Gi
                (upper_limit ≥ r[idx] ≥ lower_limit) && continue
                delete!(G[i], idx)
            end

        elseif ri == R
            upper_limit, lower_limit = R, rlayer[2]

            for idx in Gi
                (r[idx] < lower_limit) && delete!(G[i], idx)
            end

        elseif ri == rlayer[end]
            upper_limit, lower_limit = rlayer[end-1], 0f0

            for idx in Gi
                (r[idx] > upper_limit) && delete!(G[i], idx)
            end

        elseif ri ∈ @views rlayer[2:end-1]
            idx = findall(ri .== rlayer)[1]
            upper_limit, lower_limit = rlayer[idx-1], rlayer[idx+1]

            for idx in Gi
                (upper_limit ≥ r[idx] ≥ lower_limit) && continue
                delete!(G[i], idx)
            end

        end
    end

end

function find_layer(ri, rlayer)
    ri < rlayer[end] && return (rlayer[end], 0f0)
    for i in 1:length(rlayer)-1
        if rlayer[i] > ri > rlayer[i+1]
            return (rlayer[i], rlayer[i+1])
        end
    end
end

function graph2sparse(G)
    # Convert it to sparse array (better for GPU)
    I, J, V = Int[], Int[], Bool[]
    for (i, Gi) in G
         for Gj in Gi
            push!(J, i) # nodes go along columns (CSC format)
            push!(I, Gj)
            push!(V, true)
        end
    end
    K = sparse(I, J, V)
end

function cleanse_graph!(G)
    for (i, Gi) in G
        i ∈ Gi && delete!(G[i], i)
    end
end

function element_neighbours(e2n)    
    # els = size(e2n,1) == 3 ? e2n : view(e2n, 1:3,:)
    nel = length(e2n)

    # Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    @inbounds for i in eachindex(e2n)
        element = e2n[i]
        for j in eachindex(element)
            node = element[j]
            push!(I, node)
            push!(J, i)
            push!(V, true)
        end
    end
    incidence_matrix = sparse(J, I, V)

    # Find neighbouring elements
    neighbour = [Int64[] for _ in 1:nel]
    @inbounds for node in 1:nel
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

function map_edges(e2n, i, map_triangle, map_rectangle)
    element = e2n[i]
    local_map = length(element) == 4 ? map_rectangle : map_triangle
    e2n[i][local_map]
end

function edge_connectivity(gr)
    e2n, neighbours = gr.e2n, gr.neighbours
    nel = gr.nel

    map_triangle = [ 1 2 3
                     2 3 1]
    map_rectangle = [ 1 2 3 4
                      2 3 4 1]
    global_idx = 0
    el2edge = fill(0, 4, nel)
    edge2node = Vector{Int64}[]
    edge_neighbour = Matrix{Int64}(undef, 2, 4)

    edge2el = Dict{Int, Set{Int}}()
    @inbounds for iel in 1:nel
        element = e2n[iel]
        # local_map = length(element) == 4 ? map_rectangle : map_triangle
        # # local edge
        # edge = element[local_map] 
        edge = map_edges(e2n, iel, map_triangle, map_rectangle)
        sort!(edge, dims=1)
        
        # neighbours of local element
        el_neighbours = neighbours[iel]
        edge_neighbours = [map_edges(e2n, i, map_triangle, map_rectangle) for i in el_neighbours]

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

function secondary_nodes(gr; spacing = 20)
    
    e2n, θ, r = gr.e2n, gr.θ, gr.r
    neighbours = gr.neighbours

    _, edges2node, edge2el = edge_connectivity(gr)
    nedges = length(edges2node)
    # make mid points
    nnods = length(edges2node) * 40
    θmid = Vector{Float64}(undef, nnods)
    rmid = similar(θmid)
    ϵ =  2π-(1-1/gr.nθ)
    icenter = gr.nr*gr.nθ + 1
    center = closest_point(gr, 0f0, 0f0; system = :polar) # id of node at the center of the universe

    mid2nodes = Dict{Int, NTuple{2, Int}}()
    
    # main loop
    global_idx = 0
    nnods0 = length(r)
    L = 0f0
    for i in 1:nedges
        idx = edges2node[i]
        #edge coordinates
        θbar = (θ[edges2node[i][1]], θ[edges2node[i][2]])
        rbar = (r[edges2node[i][1]], r[edges2node[i][2]])

        # correct angle of nodes at the center or if θ2-θ1 > π
        θbar = correct_θ(θbar, icenter, idx, ϵ)
        # edge length
        L = edge_length(θbar, rbar)
        # number of points per edge
        npoints = Int(L ÷ spacing)
        # npoints = 9
        # add npoints to i-th edge
        if npoints > 0
            for j in 1:npoints
                # global index
                global_idx += 1
                # edge lenght
                Lθ = (θbar[2]-θbar[1])
                Lr = (rbar[2]-rbar[1])
                # increment from 1st node of the edge
                Δθ = Lθ * j / (npoints+1)
                Δr = Lr * j / (npoints+1)
                # new coordinate
                θmid[global_idx] = θbar[1] + Δθ
                rmid[global_idx] = rbar[1] + Δr
                # add node to connectivity matrix
                for iel in edge2el[i]
                    push!(gr.e2n[iel], global_idx+nnods0)

                    # add node to 2 levels of adjacency
                    for i1 in neighbours[iel]
                        if global_idx+nnods0 ∉ gr.e2n[i1]
                            push!(gr.e2n[i1], global_idx+nnods0)
                        end
                        # for i2 in neighbours[i1]
                        #     if global_idx ∉ gr.e2n[i2]
                        #         push!(gr.e2n[i2], global_idx)
                        #     end
                        # end
                    end
                end
                # update dictionary
                mid2nodes[global_idx+nnods0] = idx
            end
        end

    end

    θnew, rnew = vcat(θ, θmid[1:global_idx]), vcat(r, rmid[1:global_idx])
    x, z = @cartesian(θnew, rnew)

    gr = Grid2D(
        x,
        z,
        θnew,
        rnew,
        e2n,
        gr.nθ,
        gr.nr,
        gr.nel,
        length(x),
        gr.neighbours,
        gr.element_type
    )

    # G = nodal_incidence(gr, star_levels =1)

    # # expand adjacency list of edge nodes 
    # # @inbounds 
    # for (imid, ibar) in mid2nodes
    #     if  center ∉ G[ibar[1]]
    #         union!(G[imid], G[ibar[1]])
    #     end
    #     if  center ∉ G[ibar[2]]
    #         union!(G[imid], G[ibar[2]])
    #     end
    # end

    return gr #, G
end

function edge_length(θbar, rbar)
    θbar[1] == θbar[2] && return polardistance(θbar[1], θbar[2], rbar[1], rbar[2]) 
    # otherwise use arc length
    return arclength(θbar[1], θbar[2], rbar[1])
end

polardistance(θ1, θ2, r1, r2) = √(r1^2+r2^2-2*r1*r2*cos(θ1-θ2))

arclength(θ1, θ2, r) = r*abs(θ2-θ1)

@inbounds function correct_θ(θbar, icenter, idx, ϵ)
    if icenter ∉ idx
        if abs(θbar[1] - θbar[2]) >= ϵ
            if θbar[1] < π
                θbar = (θbar[1]+2π, θbar[2]) 
            
            elseif θbar[2] < π
                θbar = (θbar[1], θbar[2]+2π)
            end
        end
    else
        θmax = maximum(θbar)
        θbar = (θmax, θmax)
    end
    return θbar
end

function point_ids(M::Grid2D)

    top = "outter"
    bot = "inner"
    inner = "inside"

    nnod = length(M.r)
    IDs = Vector{String}(undef,nnod)

    rmin, rmax = extrema(M.r)
    
    @inbounds for (i, ri) in enumerate(M.r)
        if ri == rmax
            IDs[i] = top
        elseif ri == rmin
            IDs[i] = bot
        else
            IDs[i] = inner
        end
    end

    return IDs
end

function nodal_incidence(gr::Grid2D; star_levels = 1)
    connectivity_matrix = gr.e2n
    nels = gr.nel
    nnods = length(gr.x)
    Q = Dict{Int, Set{Int}}() # 26 = max num of neighbours
    N = zeros(Int, nnods) # num of neighbours found

    for i in 1:nels
        element_nodes = connectivity_matrix[i]
        for nJ in element_nodes
            if !haskey(Q, nJ)
                Q[nJ] = Set{Int64}()
            end
            Qi = Q[nJ]
            for nI in element_nodes
                if nI != nJ && nI ∉ Qi
                    # update helpers
                    N[nJ] += 1
                    push!(Q[nJ], nI)
                end
            end
        end
    end
    
    return Q
end

distance(ax::Real, az::Real, bx::Real, bz::Real) = √((ax-bx)^2 + (az-bz)^2)

distance(a::Point2D{Cartesian}, b::Point2D{Cartesian}) = √((a.x-b.x)^2 + (a.z-b.z)^2)

function closest_point(gr, px, pz; system = :cartesian)
    n = length(gr.x)
    dist = Inf
    di = 0.0
    index = -1
    for i in 1:n

        if system == :cartesian
            @inbounds di = distance(gr.x[i], gr.z[i], px, pz)
        
        elseif system == :polar
            @inbounds di = distance(gr.θ[i], gr.r[i], px, pz)
        
        end

        if di < dist
            index = i
            dist = di
        end

    end
    return index
end

function circle(N, r; pop_end = true) 
    t = LinRange(0, Float32(2π), N)
    x = @. r*sin(t)
    z = @. r*cos(t)
    if pop_end
        pop!(x)
        pop!(z)
    end
    return x, z
end