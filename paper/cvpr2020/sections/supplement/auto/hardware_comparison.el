(TeX-add-style-hook
 "hardware_comparison"
 (lambda ()
   (TeX-run-style-hooks
    "sections/figures/captured/midas/8_29_kitchen_scene/rmses"
    "sections/figures/captured/midas/8_29_conference_room_scene/rmses"
    "sections/figures/captured/midas/8_30_conference_room2_scene/rmses"
    "sections/figures/captured/midas/8_30_Hallway/rmses"
    "sections/figures/captured/midas/8_30_poster_scene/rmses"
    "sections/figures/captured/midas/8_30_small_lab_scene/rmses"
    "sections/figures/captured/midas/8_31_outdoor3/rmses"
    "sections/figures/captured/densedepth/8_29_kitchen_scene/rmses"
    "sections/figures/captured/densedepth/8_29_conference_room_scene/rmses"
    "sections/figures/captured/densedepth/8_30_conference_room2_scene/rmses"
    "sections/figures/captured/densedepth/8_30_Hallway/rmses"
    "sections/figures/captured/densedepth/8_30_poster_scene/rmses"
    "sections/figures/captured/densedepth/8_30_small_lab_scene/rmses"
    "sections/figures/captured/densedepth/8_31_outdoor3/rmses"
    "sections/figures/captured/dorn/8_29_kitchen_scene/rmses"
    "sections/figures/captured/dorn/8_29_conference_room_scene/rmses"
    "sections/figures/captured/dorn/8_30_conference_room2_scene/rmses"
    "sections/figures/captured/dorn/8_30_Hallway/rmses"
    "sections/figures/captured/dorn/8_30_poster_scene/rmses"
    "sections/figures/captured/dorn/8_30_small_lab_scene/rmses"
    "sections/figures/captured/dorn/8_31_outdoor3/rmses")
   (LaTeX-add-labels
    "fig:midas_captured"
    "fig:midas_outdoor_captured"
    "fig:densedepth_captured"
    "fig:densedepth_outdoor_captured"
    "fig:dorn_captured_1"
    "fig:dorn_captured_2"
    "fig:dorn_outdoor_captured"))
 :latex)

