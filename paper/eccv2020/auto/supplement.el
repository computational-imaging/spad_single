(TeX-add-style-hook
 "supplement"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("llncs" "runningheads")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "width=122mm" "left=12mm" "paperwidth=146mm" "height=193mm" "top=12mm" "paperheight=217mm") ("inputenc" "utf8x")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "mark_defs"
    "sections/supplement/simulation_comparison"
    "sections/supplement/hardware_comparison"
    "llncs"
    "llncs10"
    "graphicx"
    "comment"
    "amsmath"
    "amssymb"
    "color"
    "ruler"
    "geometry"
    "booktabs"
    "caption"
    "array"
    "tabularx"
    "bm"
    "multirow"
    "float"
    "inputenc"
    "algorithm"
    "algpseudocode"
    "algorithmicx"
    "dblfloatfix")
   (TeX-add-symbols
    '("edit" 1)
    "eg"
    "Eg"
    "ie"
    "Ie"
    "cf"
    "Cf"
    "etc"
    "vs"
    "wrt"
    "dof"
    "etal"
    "ECCVSubNumber")
   (LaTeX-add-labels
    "fig:hardware"
    "fig:comparison"
    "tab:photon_counts"
    "fig:sbr_calculation"
    "fig:sid_ablation"
    "alg:ehm"
    "alg:move_pixels"
    "fig:dither")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

