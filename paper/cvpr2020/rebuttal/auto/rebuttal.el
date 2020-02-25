(TeX-add-style-hook
 "rebuttal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "pagebackref=true" "breaklinks=true" "letterpaper=true" "colorlinks" "bookmarks=false")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "cvpr"
    "times"
    "epsfig"
    "graphicx"
    "amsmath"
    "amssymb"
    "float"
    "hyperref")
   (TeX-add-symbols
    "cvprPaperID"
    "httilde")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

