(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("llncs" "runningheads")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8x")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "mark_defs"
    "sections/introduction"
    "sections/related"
    "sections/method"
    "sections/simulation"
    "sections/hardware"
    "sections/discussion"
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
    "dblfloatfix")
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
    "sec:method"
    "sec:evaluation"
    "sec:prototype"
    "sec:discussion")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

