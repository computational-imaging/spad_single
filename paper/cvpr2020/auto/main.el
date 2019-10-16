(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8x") ("hyperref" "pagebackref=true" "breaklinks=true" "letterpaper=true" "colorlinks" "bookmarks=false")))
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
    "sections/introduction"
    "sections/related"
    "sections/method"
    "sections/simulation"
    "sections/hardware"
    "sections/discussion"
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
    "sec:intro"
    "sec:related"
    "sec:method"
    "sec:evaluation"
    "sec:prototype"
    "sec:discussion")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

