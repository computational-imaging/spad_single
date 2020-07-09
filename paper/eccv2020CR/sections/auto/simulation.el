(TeX-add-style-hook
 "simulation"
 (lambda ()
   (TeX-run-style-hooks
    "sections/figures/sim_table")
   (LaTeX-add-labels
    "tab:comparison"
    "fig:results_simulated"))
 :latex)

