(TeX-add-style-hook
 "eccv2020submission"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("llncs" "runningheads")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8x")))
   (TeX-run-style-hooks
    "latex2e"
    "mark_defs"
    "sections/introduction"
    "sections/related"
    "llncs"
    "llncs10"
    "graphicx"
    "comment"
    "amsmath"
    "amssymb"
    "color"
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
    "dblfloatfix"
    "ruler")
   (TeX-add-symbols
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
    "sec:intro"
    "sec:related"
    "sec:blind"
    "table:headings"
    "sect:figures"
    "fig:example")
   (LaTeX-add-bibliographies
    "egbib")
   (LaTeX-add-index-entries
    "Lastname,Firstname"))
 :latex)

