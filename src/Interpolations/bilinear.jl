@inbounds function bilinear(vc, pc, vu)
   
    x1, x2 = vc[1].x, vc[2].x 
    (x2-x1 > π) && (x1 += 2π)
    z1, z2 = vc[1].z, vc[4].z 
    Δx21 = x2   - x1 
    Δz21 = z2   - z1 
    Δx2  = x2   - pc.x 
    Δx1  = pc.x - x1 
    Δz2  = z2   - pc.z 
    Δz1  = pc.z - z1

    vn = 1/(Δx21*Δz21)*
        (vu[1]*Δx2*Δz2 +
        vu[2]*Δx1*Δz2 +
        vu[4]*Δx2*Δz1 +
        vu[3]*Δx1*Δz1)

    return vn

end

function interpolation_quad!(V, gr, element)
    # vertice indexes
    el = @views element[1:4]
    # vertices coordinates
    vt = vertice(gr, el)
    # vertices velocity 
    vu = ntuple(i -> V[el[i]], Val{4}())
    for inod in @views element[5:end]
        # nodal velocity
        V[inod] = bilinear(vt, gr[inod], vu)
    end
end
