struct AdjencyList{T}
    G::Matrix{T}
    N::Vector{T} # nodal degree
end

function adjacency_matrix(gr::Grid2D, nθ, npoints)
    nmax = nθ*npoints*2
    e2n = gr.e2n
    nels = gr.nel
    nnods = length(gr.x)
    G = zeros(Int32, nmax, gr.nnods) # 24 = max num of neighbours
    N = zeros(Int32, nnods) # num of neighbours found

    @inbounds for i in 1:nels
        element_nodes = @views e2n[i]
        for nJ in element_nodes
            Gi = @views G[:, nJ]
            for nI in element_nodes
                if nI != nJ && nI ∉ Gi
                    # update arrays
                    N[nJ] += 1
                    G[N[nJ], nJ] = nI
                end
            end
        end
    end

    adjmtrx = AdjencyList(G, N)

    return adjmtrx

end

function adjacency_matrix!(adjmtrx, gr::Grid2D)

    e2n = gr.e2n
    nels = gr.nel
  
    @inbounds for i in 1:nels
        element_nodes = @views e2n[i]
        for nJ in element_nodes
            Gi = @views adjmtrx.G[:, nJ]
            for nI in element_nodes
                if nI != nJ && nI ∉ Gi
                    # update arrays
                    adjmtrx.N[nJ] += 1
                    adjmtrx.G[N[nJ], nJ] = nI
                end
            end
        end
    end

end

function adjacency_list(G::Dict)
    # find max. nodal degree
    deg = Int32.(nodal_degree(G))
    maxdeg = maximum(deg)
    # number of nodes
    nnods = length(G)
    # Allocate matrix
    adj_list = zeros(Int32, maxdeg, nnods) # 24 = max num of neighbours

    @inbounds for (j, Q) in G
        for (i, Qi) in enumerate(Q)
            adj_list[i, j] = Qi
        end
    end
    
    return AdjencyList(adj_list, deg)

end

struct SparseAdjencyList{T}
    list::Vector{T}
    deg::Vector{T}
    idx::Vector{T}
end

function sparse_adjacency_list(G::Dict)
    # find max. nodal degree
    deg = Int32.(nodal_degree(G))
    cumdeg = Int32.(vcat(0, cumsum(deg)))
    # number of nodes
    nnods = length(G)
    # Allocate matrix
    adj_vector = Vector{Int32}(undef, sum(deg) )
    idx0 = @. Int32(1)+cumdeg

    Threads.@threads for inode in 1:nnods
        @inbounds for (i, Gi) in enumerate(G[inode])
            adj_vector[cumdeg[inode] + i] = Gi
        end
    end
        
    return SparseAdjencyList(adj_vector, deg, idx0)

end

@inbounds function find_layer_number(ri, rlayer)
    
    ri > rlayer[1] && return 1
    ri < rlayer[end] && return length(rlayer)+1

    for i in 1:length(rlayer)-1
        if rlayer[i] > ri > rlayer[i+1]
            return i+1
        end
    end
end

struct GridPartition{dimB, dimL, T, M, N, B}
    id::Vector{T}
    rboundaries::NTuple{dimB, M}
    layers::NTuple{dimL, T}
    boundaries::NTuple{dimB, T}
    nlayers::N
    nboundaries::N
    iterator::B

    function GridPartition(id::Vector{T}, rboundaries::NTuple{N, M}) where {T, N, M}
        nboundaries = N
        nlayers = N + 1
        layers = ntuple( i -> "Layer_$i", Val(nlayers))
        boundaries = ntuple( i -> "Boundary_$i", Val(nboundaries))

        # make layer iterator
        nmax = 2*nlayers-1
        LayerIterations = Dict{Int, NTuple}()
        LayerIterations[1] = LayerIterations[nmax] = (layers[1], boundaries[1])
        for i in 2:nlayers-1
            # For convinience, the put first the boundary where the SSSP is going to be restarted
            LayerIterations[i] = (layers[i], boundaries[i-1], boundaries[i])
            LayerIterations[nmax-i+1] = (layers[i], boundaries[i], boundaries[i-1],)
        end
        LayerIterations[nlayers] = (layers[end], boundaries[end])

        new{nboundaries, nlayers, T, M, Int, Dict}(
            id, rboundaries, layers, boundaries, nlayers, nboundaries, LayerIterations
            )
    end

end

function partition_grid(gr)

    rlayer = (R-20f0, R-35f0, R-210f0, R-410f0, R-660f0, R-2740f0, R-2891.5f0)
    r = gr.r
    LayerID = Vector{String}(undef, length(r))

    Threads.@threads for i in eachindex(r)
        
        @inbounds if round(r[i], digits=2) ∉ rlayer
            LayerID[i] = string(
                "Layer_",
                find_layer_number(round(r[i], digits=2), rlayer)
            )
        
        else
            LayerID[i] = string(
                "Boundary_",
                findfirst(round(r[i], digits=2) .== rlayer)
            )
            
        end
    end

    return GridPartition(LayerID, rlayer)

end


# function readtest(G::Dict)
#     a = zeros(length(G))
#     for (i, Q) in G
#         for Qi in Q
#             a[i] = Qi
#         end
#     end
# end

# function readtest(List::AdjencyList)
#     G = List.G
#     N = List.N
#     a = zeros(length(G))
#     # tmp = zeros(size(G, 1))

#     for j in axes(G, 2)
#         for i in 1:N[j]
#             a[i] = G[i, j]
#         end
#     end
# end

# function readtest(Gsp::SparseMatrixCSC)
#     n = size(Gsp, 2)
#     a = zeros(n)
#     for j in 1:n
#         for Gi in @views Gsp.rowval[nzrange(Gsp, j)]
#             a[i] = Gi
#         end
#     end
# end

# @benchmark readtest($G)
# @benchmark readtest($A)
# @benchmark readtest($Gsp)