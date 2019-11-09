(TeX-add-style-hook
 "supplement"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8x") ("hyperref" "breaklinks=true" "bookmarks=false")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "mark_defs"
    "sections/supplement/simulation_comparison"
    "sections/supplement/hardware_comparison"
    "article"
    "art10"
    "cvpr"
    "times"
    "epsfig"
    "graphicx"
    "amsmath"
    "amssymb"
    "booktabs"
    "caption"
    "subcaption"
    "array"
    "tabularx"
    "bm"
    "multirow"
    "float"
    "inputenc"
    "algorithm"
    "algpseudocode"
    "algorithmicx"
    "dblfloatfix"
    "hyperref")
   (TeX-add-symbols
    "cvprPaperID"
    "httilde")
   (LaTeX-add-labels
    "fig:sid_ablation"
    "alg:move_pixels"
    "fig:dither")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

