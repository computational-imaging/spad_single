(TeX-add-style-hook
 "supplement"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "breaklinks=true" "bookmarks=false")))
   (TeX-run-style-hooks
    "latex2e"
    "mark_defs"
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
    "httilde"))
 :latex)

