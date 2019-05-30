(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("acmart" "acmtog" "anonymous" "timestamp" "review")))
   (TeX-run-style-hooks
    "latex2e"
    "sections/introduction"
    "sections/related"
    "sections/method"
    "sections/handsfaces"
    "sections/results"
    "sections/discussion"
    "acmart"
    "acmart10"
    "booktabs"
    "etoolbox")
   (TeX-add-symbols
    '("note" 1))
   (LaTeX-add-bibliographies
    "references"))
 :latex)

