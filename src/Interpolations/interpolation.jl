
include("barycentric.jl")
include("bilinear.jl")

function interpolate!(V, gr)
    nel, e2n, type = gr.nel, gr.e2n, gr.element_type

    for iel in 1:nel
        element = e2n[iel]

        if type[iel] == :Quad
            interpolation_quad!(V, gr, element)

        elseif type[iel] == :Tri
            interpolation_triangle!(V, gr, element)
        end
    end
end

vertice(gr, el) = [gr[i] for i in el]
