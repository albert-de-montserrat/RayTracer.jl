function nodal_degree(G::Dict{T, Set{T}}) where T
    n = length(G)
    degrees = Vector{T}(undef, n)
    Threads.@threads for i in 1:n
        @inbounds degrees[i] = length(G[i])
    end
    return degrees
end

function symrcm(adjgr::Dict, degrees::Vector{T}) where {T}
    # Initialization
    n = length(adjgr)
    ndegperm = sortperm(degrees) # sorted nodal degrees
    inR = fill(false, n) # Is a node in the result list?
    inQ = fill(false, n) # Is a node in the queue?
    F = T[]
    sizehint!(F, n)
    Q = T[] # Node queue
    sizehint!(Q, n)
    while true
        P = zero(T) # Find the next node to start from
        while !isempty(ndegperm)
            i = popfirst!(ndegperm)
            if !inR[i]
                P = i
                break
            end
        end
        if P == zero(T)
            break # That was the last node
        end
        # Now we have a node to start from: put it into the result list
        push!(F, P)
        inR[P] = true
        empty!(Q) # empty the queue
        append!(Q, adjgr[P])
        inQ[collect(adjgr[P])] .= true # put adjacent nodes in queue
        while length(Q) >= 1
            C = popfirst!(Q) # child to put into the result list
            inQ[C] = false # make note: it is not in the queue anymore
            if !inR[C]
                push!(F, C); inR[C] = true
            end
            for i in adjgr[C] # add all adjacent nodes into the queue
                if (!inR[i]) && (!inQ[i]) # contingent on not being in result/queue
                    push!(Q, i); inQ[i] = true
                end
            end
        end
    end
    return reverse(F) # reverse the result list
end

function graph2sparse(G, F)
    tmp = 1:length(G)
    # Convert it to sparse array (better for GPU)
    I, J, V = Int[], Int[], Bool[]
    for (i, Gi) in G
         for Gj in Gi
            push!(J, i) # nodes go along columns (CSC format)
            push!(I, tmp[F[Gj]])
            push!(V, true)
        end
    end
    K = sparse(I, J, V)
end

function reorder!(gr, prm) where T
    gr.x .= gr.x[prm]
    gr.z .= gr.z[prm]
    gr.θ .= gr.θ[prm]
    gr.r .= gr.r[prm]

    map = rordering_map(prm)

    # reorder connectivity
    @inbounds for i in 1:gr.nel, j in 1:length(gr.e2n[i])
        gr.e2n[i][j] = map[gr.e2n[i][j]]
    end

    # # reorder adjacency
    # Q = Set{T}()
    # G0 = deepcopy(G)
    # for i in 1:gr.nnods
    #     for Qi in G0[i]
    #         push!(Q, map[Qi])
    #     end
    #     G[i] = deepcopy(Q)
    #     empty!(Q)
    # end
end

function rordering_map(prm::Vector{T}) where T
    n = length(prm)
    map = Dict{T, T}()
    for i in 1:n
        map[prm[i]] = i
    end
    return map
end