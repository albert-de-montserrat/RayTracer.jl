function weights(G::Dict{M,Set{M}}, gr::Grid, U::Vector{T}) where {T,M}
    # G is a dictinary of the graph with vertices v and edges e.
    # each entry of G contains e = (u, v\_i) ∈ G  

    # sparse indices and values
    I, J, V = M[], M[], T[]
    for (u, e) in G
        for v in e
            push!(I, u)
            push!(J, v)
            push!(V, edge_weight(gr[u], gr[v], U[u], U[v]))
        end
    end

    # build sparse matrix of edge weights
    w = sparse(I, J, V)
    return w
end

edge_weight(p1, p2, U1, U2) = distance3D(p1, p2) * (1 / abs(U1 + U2)) * 2

# fw = edge_weight
