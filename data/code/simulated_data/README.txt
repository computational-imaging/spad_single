11/13/18

So basically what was making ConvertRGBD segfault was that the inpainted depth maps would sometimes be so bad that the DenoisePoints() function would give points with NaN values.

These values would be passed on (silently) to LLE3, which would pass them on to ANN, at which point the program would crash at the sl_midpt_split function when the NaN values were actually accessed. This made it impossible to assign a dimension to split over (for the split step) and the variable cut_dim would be left uninitialized. Indexing into the bnds.lo and bnds.hi arrays with cut_dim would trigger the segmentation error.

Mark Nishimura

