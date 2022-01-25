include("src/StructuredGrid.jl")

c0 = (0.0, 0.0, 0.0) # coordinats of corner 1
c1 = (1.0, 1.0, 1.0) # coordinats of corner opposite to corner 1
nnods = (11, 11, 11) # number of nodes per direction

gr = grid(c0, c1, nnods)

gr[1] # linear index
gr[1, 1, 1] # cartesian index

e2n = connectivity(gr) # element-node (8 nodes per element) connectivity

connectivity(gr, 1) # if you need the connectivity of a single element