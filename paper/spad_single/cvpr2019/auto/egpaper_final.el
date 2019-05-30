(TeX-add-style-hook
 "egpaper_final"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "breaklinks=true" "bookmarks=false")))
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
    "sections/results"
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
    "subfig"
    "array"
    "tabularx"
    "bm"
    "hyperref")
   (TeX-add-symbols
    "cvprPaperID"
    "httilde"))
 :latex)

