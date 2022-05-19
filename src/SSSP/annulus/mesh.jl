using Triangulate, SparseArrays

struct Delauney2D{T,M,N}
    x::Vector{M} # cartesian coordinates
    z::Vector{M} # cartesian coordinates
    θ::Vector{M} # polar coordinates
    r::Vector{M} # polar coordinates
    e2n::Matrix{T} # connectivity matrix
    neighbours_list::N # neighbouring elements
    nels::T # number of elements
    nnods::T # number of nodes
end

function triangle_annulus_2D(;
    mesh_type=:structured, Nsurf=50, Nx=20, Nz=20, npoints=1, max_area=250.0f2
)

    # Radius of the earth
    R = 6371.0f0

    # define earths boundary
    xs, zs = circle(Nsurf, R)

    # triangulate domain
    if mesh_type == :structured
        triout, = structured_convex_hull(xs, zs, Nx, Nz, R)

    elseif mesh_type == :unstructured
        triout, = unstructured_constrained_domain(xs, zs, max_area)

    elseif mesh_type == :layered
        triout, = layered_domain(xs, zs, max_area)
    end

    # unpack triangulation
    xtri = Float32.(triout.pointlist[1, :])
    ztri = Float32.(triout.pointlist[2, :])
    e2n = triout.trianglelist # connectivity matrix

    # list of neighbouring elements
    neighbours_list = element_neighbours(e2n)

    # add midpoints
    x, z, e2n = add_midpoints(e2n, neighbours_list, xtri, ztri; npoints=npoints)

    # polar coordinates
    θ, r = cart2polar(x, z)
    r = min.(R, r)

    return Delauney2D(
        x, z, θ, r, e2n, neighbours_list, Int32(size(e2n, 2)), Int32(length(x))
    )
end

function structured_convex_hull(xs, zs, Nx, Nz, R)
    # make rectangular grid (x, y) ∈ [-R, R] × [-R, R]
    x_tmp = LinRange(-R, R, Nx)
    z_tmp = LinRange(-R, R, Nz)
    xgrid = Vector{eltype(x_tmp)}(undef, Nx * Nz)
    zgrid = similar(xgrid)
    # boolean vector that tells us what nodes to keep
    ikeep = trues(Nx * Nz)
    # body loop
    @inbounds for i in 1:Nx, j in 1:Nz

        # global index
        index = i + (j - 1) * Nx

        # coordinates
        xi = x_tmp[i]
        zi = z_tmp[j]
        xgrid[index] = xi
        zgrid[index] = zi

        # keep only (x, y) ∈ [0, 2π] x [0, R]
        if distance_from_center(xi, zi) > R
            ikeep[index] = false
        end
    end

    # take nodes within the earth radius 
    x_inner = xgrid[ikeep]
    z_inner = zgrid[ikeep]

    # merge surface and inner nodes
    x = vcat(xs, x_inner)
    z = vcat(zs, z_inner)

    # triangulate convex hull
    triin = Triangulate.TriangulateIO()
    triin.pointlist = vcat(x', z')
    (triout, vorout) = triangulate("Q", triin)
    return triout, vorout
end

function unstructured_constrained_domain(xs, zs, max_area)
    # domain nodes
    Ω = vcat(xs', zs')
    # segment list of earths surface
    seglist1 = collect(1:length(xs))
    seglist2 = seglist1 .+ 1
    seglist2[end] = 1
    # triangulate
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}(Ω)
    triin.segmentlist = Matrix{Cint}(vcat(seglist1', seglist2'))
    triin.segmentmarkerlist = Vector{Int32}(seglist1)
    (triout, vorout) = triangulate("pa$(max_area)Q", triin)

    return triout, vorout
end

function layered_domain(xs, zs, max_area)
    Ns = length(xs)
    N = [Ns, Ns, Ns, Ns, Ns, Ns ÷ 2, Ns ÷ 2, Ns ÷ 10]
    cs = cumsum(N)

    depth_list = 6371.0f0 .- Float32.([20, 35, 210, 410, 660, 2740, 2891.5, 5153.5])
    nd = length(depth_list)

    # interface coordinates
    x1, z1 = circle(N[1], depth_list[1]; pop_end=true)
    x2, z2 = circle(N[2], depth_list[2]; pop_end=true)
    x3, z3 = circle(N[3], depth_list[3]; pop_end=true)
    x4, z4 = circle(N[4], depth_list[4]; pop_end=true)
    x5, z5 = circle(N[5], depth_list[5]; pop_end=true)
    x6, z6 = circle(N[6], depth_list[6]; pop_end=true)
    x7, z7 = circle(N[7], depth_list[7]; pop_end=true)
    x8, z8 = circle(N[8], depth_list[8]; pop_end=true)
    # concatenate them 
    Ω0 = vcat(xs', zs')
    Ω1 = vcat(x1', z1')
    Ω2 = vcat(x2', z2')
    Ω3 = vcat(x3', z3')
    Ω4 = vcat(x4', z4')
    Ω5 = vcat(x5', z5')
    Ω6 = vcat(x6', z6')
    Ω7 = vcat(x7', z7')
    Ω8 = vcat(x8', z8')

    # define region points
    #          xr1,          zr1, id1, elsize
    region1 = [x1[1] * 0.99, z1[1], 1, 1.0f0]
    region2 = [x2[1] * 0.99, z2[1], 2, 1.0f0]
    region3 = [x3[1] * 0.99, z3[1], 3, max_area]
    region4 = [x4[1] * 0.99, z4[1], 4, max_area]
    region5 = [x5[1] * 0.99, z5[1], 5, max_area]
    region6 = [x6[1] * 0.99, z6[1], 6, max_area]
    region7 = [x7[1] * 0.99, z7[1], 7, max_area]
    region8 = [x8[1] * 0.99, z8[1], 8, max_area]
    region_list = hcat(
        region1, region2, region3, region4, region5, region6, region7, region8
    )

    # make segment lists
    # regions 
    seglist = segment_list.(N)
    s_regions = reduce(hcat, seglist[i] .+ cs[i] for i in 1:nd)
    # surface
    s_surface = segment_list(Ns)
    s = hcat(s_surface, s_regions)

    # domain nodes
    Ω = hcat(Ω0, Ω1, Ω2, Ω3, Ω4, Ω5, Ω6, Ω7, Ω8)
    # segment list of earths surface
    # seglist1 = collect(1:length(xs))
    # seglist2 = seglist1.+1
    # seglist2[end] = 1
    # triangulate
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}(Ω)
    triin.segmentlist = Matrix{Cint}(s)
    triin.segmentmarkerlist = Vector{Int32}(s[1, :])
    triin.regionlist = Matrix{Cdouble}(region_list)
    mangle = 20.0f0
    (triout, vorout) = triangulate("pa$(max_area)Aq$(mangle)", triin)

    # xz = triout.pointlist'
    # e2n = triout.trianglelist'
    # poly(xz, e2n, strokewidth=1, shading=true, color = nothing)

    return triout, vorout
end

function segment_list(n)
    seglist1 = collect(1:n)
    seglist2 = seglist1 .+ 1
    seglist2[end] = 1
    return vcat(seglist1', seglist2')
end

function circle(N, r; pop_end=true)
    t = LinRange(0, Float32(2π), N)
    x = @. r * cos(t)
    z = @. r * sin(t)
    if pop_end
        pop!(x)
        pop!(z)
    end
    return x, z
end

function distance(ax::T, az::T, bx::T, bz::T) where {T<:Number}
    return √(muladd(az - bz, az - bz, (ax - bx) * (ax - bx)))
end

distance_from_center(x::T, z::T) where {T<:Number} = √(x * x + z * z)

cart2polar(x::T, z::T) where {T<:Real} = (atan(x, z), distance_from_center(x, z))

function cart2polar(x::Vector{T}, z::Vector{T}) where {T<:Real}
    θ = similar(x)
    r = similar(x)
    for i in eachindex(x)
        @inbounds θ[i], r[i] = cart2polar(x[i], z[i])
    end
    return θ, r
end

function element_neighbours(e2n)
    # Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    @inbounds for i in axes(e2n, 1), j in axes(e2n, 2)
        node = e2n[i, j]
        push!(I, node)
        push!(J, j)
        push!(V, true)
    end
    incidence_matrix = sparse(J, I, V)

    # Find neighbouring elements
    neighbour = Dict{Int,Vector{Int}}()
    nnod = maximum(e2n)
    @inbounds for node in 1:nnod
        # Get elements sharing node
        r = nzrange(incidence_matrix, node)
        el_neighbour = incidence_matrix.rowval[r]
        # Add neighbouring elements to neighbours map
        for iel1 in el_neighbour
            if !haskey(neighbour, iel1)
                neighbour[iel1] = Int[]
            end
            current_neighbours = neighbour[iel1]
            for iel2 in el_neighbour
                # check for non-self neighbour and avoid repetitions
                if (iel1 != iel2) && (iel2 ∉ current_neighbours)
                    push!(neighbour[iel1], iel2)
                end
            end
        end
    end

    return neighbour
end

function edge_connectivity(e2n, neighbours)
    nel = size(e2n, 2)
    local_map = [1 2 3; 2 3 1]
    global_idx = 0
    el2edge = Dict(i => zeros(Int, 3) for i in 1:nel) # fill(0,3,nel)
    edge2node = Dict{Int64,Vector{Int32}}() # Vector{Int64}[]
    edge_neighbour = Array{Int64,2}(undef, 2, 3)

    for iel in 1:nel
        # local edge
        edge = e2n[local_map, iel]
        sort!(edge; dims=1)

        # neighbours of local element
        el_neighbours = neighbours[iel]
        edge_neighbours = [e2n[local_map, i] for i in el_neighbours]

        # check edge by edge
        for iedge in 1:3
            if el2edge[iel][iedge] == 0
                global_idx += 1
                el2edge[iel][iedge] = global_idx
                edge2node[global_idx] = edge[:, iedge]

                # edges the neighbours
                for (c, ieln) in enumerate(el_neighbours)
                    edge_neighbour .= edge_neighbours[c]
                    sort!(edge_neighbour; dims=1)

                    # # check wether local edge is in neighbouring element
                    # # @show edge, edge_neighbour
                    # for i in 1:3 
                    #     if issubset(edge[iedge], view(edge_neighbour, :,i))
                    #         el2edge[ieln][i] = global_idx
                    #         break
                    #     end
                    # end

                    # check wether local edge is in neighbouring element
                    for i in 1:3
                        if (edge[1, iedge] == edge_neighbour[1, i]) &&
                            (edge[2, iedge] ∈ edge_neighbour[2, i])
                            el2edge[ieln][i] = global_idx
                            break
                        end
                    end
                end
            end
        end
    end

    return el2edge, edge2node
end

function add_midpoints(
    e2n, neighbours_list, x::Vector{T}, z::Vector{T}; npoints=1
) where {T}
    el2edge, edges2node = edge_connectivity(e2n, neighbours_list)

    # number of edges
    nedges = length(edges2node)

    # number of nodes to add per edge
    nnods = nedges * npoints

    # alocate edge nodes
    xmid = Vector{T}(undef, nnods)
    zmid = similar(xmid)

    ##
    # add notes to connectivity matrix
    ##
    # number of nodes to add per element
    nxel = 3 * npoints
    nel = size(e2n, 2)
    e2n_new = Matrix{eltype(e2n)}(undef, nxel, nel)
    nnod0 = length(x)

    # main loop
    @inbounds for i in 1:nedges
        #edge coordinates
        xbar = (x[edges2node[i][1]], x[edges2node[i][2]])
        zbar = (z[edges2node[i][1]], z[edges2node[i][2]])

        # add npoints to i-th edge
        for j in 1:npoints
            # global index
            index = j + (i - 1) * (npoints)
            # edge lenght
            Lx = (xbar[2] - xbar[1])
            Lz = (zbar[2] - zbar[1])
            # increment from 1st node of the edge
            Δx = Lx * j / (npoints + 1)
            Δz = Lz * j / (npoints + 1)
            # new coordinate
            xmid[index] = xbar[1] + Δx
            zmid[index] = zbar[1] + Δz
        end
    end

    ##
    # add notes to connectivity matrix
    ##
    # number of nodes to add per element
    nxel = 3 * npoints
    nel = size(e2n, 2)
    e2n_new = Matrix{eltype(e2n)}(undef, nxel, nel)
    nnod0 = length(x)

    for iel in 1:nel
        element_edges = el2edge[iel]

        for (j, iedge) in enumerate(element_edges)
            # global indices of edge nodes
            nodal_indices = (iedge - 1) * npoints + nnod0 + 1
            index_range = nodal_indices:(nodal_indices + npoints - 1)
            # indices of the new connectivity matrix to be updated
            local_indices = (j:(j + (npoints - 1))) .+ (npoints - 1) * (j - 1)
            # update connectivity matrix
            e2n_new[local_indices, iel] .= index_range
        end
    end

    x, z = vcat(x, xmid), vcat(z, zmid)
    e2n = vcat(e2n, e2n_new)

    return x, z, e2n
end

function nodal_incidence(gr::Delauney2D)

    # unpack from grid structure
    connectivity_matrix = gr.e2n
    T = eltype(connectivity_matrix)
    nels = gr.nels
    nnods = gr.nnods

    # Dictionary containing adjacent neighbours 
    Q = Dict{T,Set{T}}()
    # num of adjacent nodes of i-th node
    N = zeros(Int, nnods)

    @inbounds for i in 1:nels
        element_nodes = @views connectivity_matrix[:, i]
        for nJ in element_nodes
            if !haskey(Q, nJ)
                Q[nJ] = Set{T}()
            end
            Qi = Q[nJ]
            for nI in element_nodes
                if (nI != nJ) && (nI ∉ Qi)
                    # update helpers
                    N[nJ] += 1
                    push!(Q[nJ], nI)
                end
            end
        end
    end

    # Convert it to sparse array (better for GPU)
    I, J, V = Int[], Int[], Bool[]
    for (i, Qi) in Q
        for Qj in Qi
            push!(J, i) # nodes go along columns (CSC format)
            push!(I, Qj)
            push!(V, true)
        end
    end
    K = sparse(I, J, V)

    return Q, K
end
