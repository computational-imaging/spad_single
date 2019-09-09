(TeX-add-style-hook
 "egpaper_final"
 (lambda ()
<<<<<<< HEAD
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "breaklinks=true" "bookmarks=false")))
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
    "httilde"))
=======
   (LaTeX-add-bibitems
    "Rother2006"
    "Eigen2014"
    "Laina2016"
    "Xu2017"
    "Godard2017"
    "Fu2018"
    "Hao2018"
    "Xu2018"
    "Alhashim2018"
    "Gonzalez2008"
    "Hoiem2005"
    "Horaud2016"
    "Karsch2014"
    "Lamb2010"
    "Lindell2018"
    "Morovic2002"
    "Niclass2005"
    "Nikolova2013"
    "Stoppa2007"
    "Swoboda2013"
    "Veerappan2011"
    "Saxena2006"
    "Zhang2018"
    "Zhang2017"))
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
 :latex)

