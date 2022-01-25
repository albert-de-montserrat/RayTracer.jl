
function sparse_adjacency_list(IM::SparseMatrixCSC{Bool, Int64}, gr)
    (; e2n, nnods) = gr
    hint = maximum([length(e) for (_, e) in e2n]) ÷ 2
    N = zeros(Int16, nnods)
    for node in 1:IM.n
        @inbounds for iel in @views IM.rowval[nzrange(IM, node)]    
            N[node] += length(e2n[iel])
        end
    end

    
    Q = Dict{Int, Vector{Int}}() # 26 = max num of neighbours
    for node in 1:IM.n
        @inbounds for iel in @views IM.rowval[nzrange(IM, node)]    
            if !haskey(Q, node)
                # Q[node] = Set{Int64}()
                Q[node] = e2n[iel]
                # sizehint!(Q[node], N[node])
            end
            Q[node] = vcat(Q[node], e2n[iel])
        end
    end

    # Q

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
            # LayerIterations[nmax-i+1] = (layers[i], boundaries[i], boundaries[i-1],) # original
            LayerIterations[nmax-i+1] = (layers[i], boundaries[i-1], boundaries[i]) 
        end
        LayerIterations[nlayers] = (layers[end], boundaries[end])

        new{nboundaries, nlayers, T, M, Int, Dict}(
            id, rboundaries, layers, boundaries, nlayers, nboundaries, LayerIterations
            )
    end

end

function partition_grid(gr; rlayer = (R-20f0, R-35f0, R-210f0, R-410f0, R-660f0, R-2740f0, R-2891.5f0))

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

struct VelocityDiscontinuity{NB}
    depth::NTuple{NB, Real}
end

```
    check whether r is inside two velocity boundaries
```
macro inlayer(r, rlayer)
    quote 
        @inbounds for i in 1:length($rlayer)+1
            if i == 1
                $r > $rlayer[i] && return i

            elseif i == (length($rlayer)+1)
                $r < $rlayer[length($rlayer)] && return i

            else
                $rlayer[(i-1)] > $r > $rlayer[i]  && return i
            end
        end
        return nothing
    end
end

```
    check whether r is laying on a boundary
```
macro inboundary(r, rlayer)
    quote 
        for i in 1:length($rlayer)
            $r ∈ $rlayer[i] && return i
        end
        return nothing
    end
end

struct VelocityModel{N, NB, NL}
    source_boundary::Integer
    depth_boundary::NTuple{N, Real}
    phase::NTuple{NL, Symbol}
    active_layers::NTuple{NL, Integer}
    active_boundaries::NTuple{NB, Integer}
    boundary_model::NTuple{NB, Symbol}

    function VelocityModel(
        source_boundary::Integer,
        depth_boundary::NTuple{N, Real},
        active_boundaries::NTuple{NB, Int}, 
        phase::NTuple{NL, Symbol}, 
        boundary_model::NTuple{NB, Symbol}
        ) where {N, NL, NB}
    
        active_layers = ntuple(i->active_boundaries[i], Val{NB}())
        VelocityModel{N, NB, NL}(
            source_boundary,
            depth_boundary,
            phase,
            active_layers,
            active_boundaries,
            boundary_model,
        )
    end

end

# rl = (R-20f0, R-35f0, R-210f0, R-410f0, R-660f0, R-2740f0, R-2891.5f0)
# phase = (:P,:P,:S)
# active_layers = (1,2,3)
# active_boundaries = (1,2,4)
# boundary_model = (:transmited, :transmited, :reflected)
# source_boundary = 0

# VelocityModel(
#     source_boundary,
#     rl,
#     active_boundaries,
#     phase,
#     boundary_model,
#     )
