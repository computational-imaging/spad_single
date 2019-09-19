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
    "sections/figures/captured/densedepth/8_29_kitchen_scene/rmses"
    "sections/figures/captured/densedepth/8_29_conference_room_scene/rmses"
    "sections/figures/captured/midas/8_30_conference_room2_scene/rmses"
    "sections/figures/captured/densedepth/8_30_Hallway/rmses"
    "sections/figures/captured/densedepth/8_30_poster_scene/rmses"
    "sections/figures/captured/densedepth/8_30_small_lab_scene/rmses"
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
   (LaTeX-add-labels
    "fig:midas_captured"))
 :latex)

