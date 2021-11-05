import Base:getindex

abstract type Mesh end
abstract type QuadraticMesh <:Mesh end
abstract type LinearMesh <:Mesh end

struct Grid{T, M, N} 
    c0::NTuple{3, M} # origin corner
    c1::NTuple{3, M} # opposite corner
    nels::NTuple{3, N} # number of elements
    nnods::NTuple{3, N} # number of nodes
    nxny::Int64
    x::Vector{M} # nodal ranges
    y::Vector{M} # nodal ranges
    z::Vector{M} # nodal ranges
end

struct LazyGrid{M, N} 
    c0::NTuple{3, M} # origin corner
    c1::NTuple{3, M} # opposite corner
    Δ::NTuple{3, M} # directional steps
    nels::NTuple{3, N} # number of elements
    nnods::NTuple{3, N} # number of nodes
    nxny::Int64
end

struct Point{T}
    x::T
    y::T
    z::T
end

Π(n::NTuple{N, T}) where {N, T} = foldl(*, n)

function grid(c0, c1, nnods, T::Type{LinearMesh})
    nels = nnods.-1
    
    x = collect(LinRange(c0[1], c1[1], nnods[1]))
    y = collect(LinRange(c0[2], c1[2], nnods[2]))
    z = collect(LinRange(c0[3], c1[3], nnods[3]))
    
    Grid{T, eltype(c0), eltype(nels)}(
        c0,
        c1,
        nels,
        nnods,
        nnods[1]*nnods[2],
        x,
        y,
        z,
    )

end

function lazy_grid(c0, c1, nnods)
    nels = nnods.-1
    
    Δx = (c1[1] - c0[1])/(nnods[1]-1)
    Δy = (c1[2] - c0[2])/(nnods[2]-1)
    Δz = (c1[3] - c0[3])/(nnods[3]-1)
    
    LazyGrid(
        c0,
        c1,
        (Δx, Δy, Δz),
        nels,
        nnods,
        nnods[1]*nnods[2],
    )

end

function getindex(gr::Grid, I::Int, J::Int, K::Int)
    @assert I <= gr.nnods[1]
    @assert J <= gr.nnods[2]
    @assert K <= gr.nnods[3]
    Point(gr.x[I], gr.y[J], gr.z[K])
end

function getindex(gr::LazyGrid, I::Int, J::Int, K::Int)
    @assert I <= gr.nnods[1]
    @assert J <= gr.nnods[2]
    @assert K <= gr.nnods[3]
    Point(
        gr.c0[1] + (I-1)*gr.Δ[1],
        gr.c0[2] + (J-1)*gr.Δ[2],
        gr.c0[3] + (K-1)*gr.Δ[3],
    )
end

function getindex(gr::Grid, I::Int)
    # @assert I <= Π(gr.nnods)
    i,j,k = CartesianIndex(gr, I)
    Point(gr.x[i], gr.y[j], gr.z[k])
end

function getindex(gr::LazyGrid, I::Int)
    @assert I <= Π(gr.nnods)
    i,j,k = CartesianIndex(gr, I)
    gr[i,j,k]
end

getindex(gr::Grid, ::NTuple{N, T}) where {N, T} = [gr[i] for i in 1:N]   
getindex(gr::LazyGrid, ::NTuple{N, T}) where {N, T} = [gr[i] for i in 1:N]

function CartesianIndex(gr::Grid, I::Int)
    nx = gr.nnods[1]
    i = mod(I-1, nx)+1
    k = div(I, gr.nxny, RoundUp)
    j = div(I-gr.nxny*(k-1), nx, RoundUp)
    (i, j, k)
end

function CartesianIndex(gr::LazyGrid, I::Int)
    nx = gr.nnods[1]
    i = mod(I-1, nx)+1
    k = div(I, gr.nxny, RoundUp)
    j = div(I-gr.nxny*(k-1), nx, RoundUp)
    (i, j, k)
end

# lower-front-left corner nodal index of I-th elements
cornerindex_ijk(gr::Grid{LinearMesh,M,N}, iel::Int)  where {M, N} = (
    mod(iel-1, gr.nels[1])+1,
    mod(div(iel, gr.nels[1], RoundUp)-1, gr.nels[2])+1,
    div(iel, gr.nels[1]*gr.nels[2], RoundUp),
)

function cornerindex(gr::Grid{LinearMesh,M,N}, iel::Int) where {M, N}
    nx, ny, = gr.nnods
    i, j, k = cornerindex_ijk(gr, iel)
    i + (j-1)*nx + (k-1)*nx*ny 
end

function connectivity(gr::Grid{LinearMesh,M,N}) where {M, N} 
    #=
        8_____________9
       /|            /|       
      / |           / |   
    5/__|_________6/  |
     |  |          |  | 
     |  4__________|_3|  
     |  /          |  /
     | /           | /
    1|/___________2|/

    =#
    nel = Π(gr.nels)
    e2n = Vector{NTuple{8,N}}(undef, nel)
    Threads.@threads for iel in 1:nel
        @inbounds e2n[iel] = connectivity(gr, iel) 
    end
    return e2n
end

function connectivity(gr::Grid{LinearMesh,M,N}, iel::Int) where {M, N}
    nx, ny = gr.nnods[1], gr.nnods[2]
    idx = cornerindex(gr, iel)
    #=
        8_____________9
       /|            /|       
      / |           / |   
    5/__|_________6/  |
     |  |          |  | 
     |  4__________|_3|  
     |  /          |  /
     | /           | /
    1|/___________2|/

    =#

    (
        idx,
        idx + 1,
        idx + 1 +nx,
        idx + nx,
        idx + nx*ny,
        idx + nx*ny + 1,
        idx + nx*ny + 1 + nx,
        idx + nx*ny + nx,
    )
end

distance(a::Point, b::Point) = √( (a.x-b.x)^2 + (a.y-b.y)^2 + (a.z-b.z)^2)

struct NodalIncidence{T, M}
    Q::T # Incidence matrix
    N::M # Number of neighbours
end

function nodal_incidence(gr; neighbour_levels = 1)
    connectivity_matrix = connectivity(gr)
    nels = Π(gr.nels)

    Q = Dict{Int, Set{Int}}() # 26 = max num of neighbours
    N = zeros(Int, Π(gr.nnods)) # num of neighbours found

    for i in 1:nels
        element_nodes = connectivity_matrix[i]
        for nJ in element_nodes
            # Qi = view(Q, nJ, :)
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

    # expand nodal incidence by neighbour_levels - levels
    for _ in 1:neighbour_levels
        Q0 = deepcopy(Q)
        for i in 1:length(N)
            for n in Q0[i]
                union!(Q[i], Q0[n])
            end
        end
    end
    # # Convert it to sparse array (better for GPU)
    # I, J, V = Int[], Int[], Bool[]
    # for (i, Qi) in Q
    #      for Qj in Qi
    #         push!(J, i) # nodes go along columns (CSC format)
    #         push!(I, Qj)
    #         push!(V, true)
    #     end
    # end
    # K = sparse(I, J, V)
    
    return Q #, K
end

function spherical2cart(θ::T, ϕ::T, r::T) where T<:Real
    x = r * cos(ϕ) * sin(θ)
    y = r * sin(ϕ) * sin(θ)
    z = r * cos(θ)
    return x, y, z
end

function spherical2cart(p::Point{T}) where T 
    x,y,z = spherical2cart(p.x, p.y, p.z)
    return Point{T}(x,y,z)
end

spherical2cart(gr::Grid) = [spherical2cart(gr[i]) for i in 1:Π(gr.nnods)]

distance3D(ax::T, ay::T, az::T, bx::T, by::T, bz::T) where T<:Real = √((ax-bx)^2 + (ay-by)^2 +(az-bz)^2)

distance3D(p1::Point, p2::Point) = distance3D(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z)

function polardistance3D(p1::Point, p2::Point)
    p1c = spherical2cart(p1)
    p2c = spherical2cart(p2)
    return distance3D(p1c, p2c)
end

function polardistance3D(aθ::T, aϕ::T, ar::T, bθ::T, bϕ::T, br::T) where T<:Real
    ax, ay, az = spherical2cart(aθ, aϕ, ar)
    bx, by, bz = spherical2cart(bθ, bϕ, br)
    return distance3D(ax, ay, az, bx, by, bz)
end

function closest_point(gr, x::T, y::T, z::T) where T
    n = Π(gr.nnods)
    p = Point{T}(x, y, z)
    dist = typemax(T)
    index = -1
    for i in 1:n
        di = distance3D(gr[i], p)
        if di < dist
            index = i
            dist = di
        end
    end
    return index
end