(TeX-add-style-hook
 "egpaper_final"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "breaklinks=true" "bookmarks=false")))
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
    "hyperref")
   (TeX-add-symbols
    "cvprPaperID"
    "httilde")
   (LaTeX-add-bibliographies
    "bib"))
 :latex)

