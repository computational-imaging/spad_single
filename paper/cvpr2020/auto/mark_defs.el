(TeX-add-style-hook
 "mark_defs"
 (lambda ()
   (TeX-run-style-hooks
    "mathtools")
   (TeX-add-symbols
    '("ord" 1)
    '("smat" 1)
    '("msset" 2)
    '("mset" 2)
    '("sbs" 1)
    '("s" 1)
    '("txt" 1)
    '("eqnsplit" 1)
    '("eqn" 1)
    '("alg" 1)
    "dom"
    "id"
    "divs"
    "tdivs"
    "ndivs"
    "N"
    "Z"
    "Q"
    "R"
    "F"
    "E"
    "X"
    "del"
    "Real"
    "Imag"
    "res")
   (LaTeX-add-environments
    "thm"
    "lem"
    "prop"
    "cor"
    "defn"
    "conj")
   (LaTeX-add-mathtools-DeclarePairedDelimiters
    '("paren" "")
    '("bracket" "")
    '("ang" "")
    '("abs" "")
    '("set" "")
    '("norm" "")))
 :latex)

