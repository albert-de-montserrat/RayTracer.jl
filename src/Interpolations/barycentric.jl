@inbounds function barycentric_coordinates(vc, pc)
    x1,x2,x3 = vc[1].x, vc[2].x, vc[3].x
    z1,z2,z3 = vc[1].z, vc[2].z, vc[3].z
    x, z      = pc.x, pc.z

    N1 = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))

    N2 = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    
    N3 = 1 - N1 - N2

    return (N1, N2, N3)

end

(⋅)(a::NTuple{3, T}, b::NTuple{3, T}) where T = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

function interpolation_triangle!(V, gr, element)
    # vertice indexes
    el = @views element[1:3]
    # vertices coordinates
    vt = vertice(gr, el)
    # vertices velocity 
    vu = ntuple(i -> V[el[i]], Val{3}())
    for inod in  @views element[4:end]
        # barycentric coordinates (i.e. interpolation weights)
        N = barycentric_coordinates(vt, gr[inod])
        # nodal velocity
        V[inod] = N⋅vu
    end
end