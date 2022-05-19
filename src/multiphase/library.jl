# struct VelocityModel{N, NB, NL}
# source_boundary::Integer
# depth_boundary::NTuple{N, Real}
# phase::NTuple{NL, Symbol}
# active_layers::NTuple{NL, Integer}
# active_boundaries::NTuple{NB, Integer}
# boundary_model::NTuple{NB, Symbol}

function model_libray(
    model;
    source_boundary=nothing,
    depth_boundary=nothing,
    phase=nothing,
    active_layers=nothing,
    active_boundaries=nothing,
    boundary_model=nothing,
)
    if model ∈ (:Pdiff, :Sdiff)
        NB = Val{length(source_boundary)}()
        NL = Val{length(source_boundary)}()
        phase = model == :Pdif ? :P : :S
        VelocityModel(
            0, #source_boundary,
            depth_boundary,
            ntuple(i -> phase, NB),
            ntuple(i -> i, NL),
            ntuple(i -> i, NB),
            ntuple(i -> :transmited, NB),
        )
    end
end
