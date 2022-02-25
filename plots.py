from dataclasses import dataclass


@dataclass
class Prob_color:
    lower_bound: int
    upper_bound: int
    color: int

# Color tables for image outputs
# Fixed, not parameters
date_color = 0XFFFFFF  # white
seed_color = 0XF9D16E  # pale yellow, urban color during test mode
water_color = 0X1434D6  # royal blue, for hillshade 0 pixels

# Urban colors during predict mode
self.probability_color_count = 0
probability_color = [
    Prob_color(0, 50, None),
    Prob_color(50, 60, 0X005A00),  # dark green
    Prob_color(60, 70, 0X008200),
    Prob_color(70, 80, 0X00AA00),
    Prob_color(80, 90, 0X00D200),
    Prob_color(90, 95, 0X00D200),  # light green
    Prob_color(95, 100, 0X8B0000),  # dark red
]

# Land cover class dictionary
# class-value-color
#  C. LAND COVER COLORTABLE
#  Land cover input images should be in grayscale GIF image format.
#  The 'pix' value indicates a land class grayscale pixel value in
#  the image.
#    pix: input land class pixel value
#    name: text string indicating land class
#    flag: special case land classes
#          URB - urban class (area is included in urban input data
#                and will not be transitioned by deltatron)
#          UNC - unclass (NODATA areas in image)
#          EXC - excluded (land class will be ignored by deltatron)
#    hex/rgb: hexidecimal or rgb (red, green, blue) output colors
#
#              pix, name,     flag,   hex/rgb, #comment

# Deltatron (Land Cover)
landuse_class: []

LANDUSE_CLASS=  0,  Unclass , UNC   , 0X000000
LANDUSE_CLASS=  1,  Urban   , URB   , 0X8b2323 #dark red
LANDUSE_CLASS=  2,  Agric   ,       , 0Xffec8b #pale yellow
LANDUSE_CLASS=  3,  Range   ,       , 0Xee9a49 #tan
LANDUSE_CLASS=  4,  Forest  ,       , 0X006400
LANDUSE_CLASS=  5,  Water   , EXC   , 0X104e8b
LANDUSE_CLASS=  6,  Wetland ,       , 0X483d8b
LANDUSE_CLASS=  7,  Barren  ,       , 0Xeec591

#  Simplify either to generate all images, or a fixed number
#  D. GROWTH TYPE IMAGE OUTPUT CONTROL AND COLORTABLE
#
#  From here you can control the output of the Z grid
#  (urban growth) just after it is returned from the spr_spread()
#  function. In this way it is possible to see the different types
#  of growth that have occured for a particular growth cycle.
#
#  VIEW_GROWTH_TYPES(YES/NO) provides an on/off
#  toggle to control whether the images are generated.
#
#  GROWTH_TYPE_PRINT_WINDOW provides a print window
#  to control the amount of images created.
#  format:  <start_run>,<end_run>,<start_monte_carlo>,
#           <end_monte_carlo>,<start_year>,<end_year>
#  for example:
#  GROWTH_TYPE_PRINT_WINDOW=run1,run2,mc1,mc2,year1,year2
#  so images are only created when
#  run1<= current run <=run2 AND
#  mc1 <= current monte carlo <= mc2 AND
#  year1 <= currrent year <= year2
#
#  0 == first
VIEW_GROWTH_TYPES(YES/NO)=NO
GROWTH_TYPE_PRINT_WINDOW=0,0,0,0,1995,2020
PHASE0G_GROWTH_COLOR=  0xff0000 # seed urban area
PHASE1G_GROWTH_COLOR=  0X00ff00 # diffusion growth
PHASE2G_GROWTH_COLOR=  0X0000ff # NOT USED
PHASE3G_GROWTH_COLOR=  0Xffff00 # breed growth
PHASE4G_GROWTH_COLOR=  0Xffffff # spread growth
PHASE5G_GROWTH_COLOR=  0X00ffff # road influenced growth

#************************************************************
#
#  E. DELTATRON AGING SECTION
#
#  From here you can control the output of the deltatron grid
#  just before they are aged
#
#  VIEW_DELTATRON_AGING(YES/NO) provides an on/off
#  toggle to control whether the images are generated.
#
#  DELTATRON_PRINT_WINDOW provides a print window
#  to control the amount of images created.
#  format:  <start_run>,<end_run>,<start_monte_carlo>,
#           <end_monte_carlo>,<start_year>,<end_year>
#  for example:
#  DELTATRON_PRINT_WINDOW=run1,run2,mc1,mc2,year1,year2
#  so images are only created when
#  run1<= current run <=run2 AND
#  mc1 <= current monte carlo <= mc2 AND
#  year1 <= currrent year <= year2
#
#  0 == first
VIEW_DELTATRON_AGING(YES/NO)=NO
DELTATRON_PRINT_WINDOW=0,0,0,0,1930,2020
DELTATRON_COLOR=  0x000000 # index 0 No or dead deltatron
DELTATRON_COLOR=  0X00FF00 # index 1 age = 1 year
DELTATRON_COLOR=  0X00D200 # index 2 age = 2 year
DELTATRON_COLOR=  0X00AA00 # index 3 age = 3 year
DELTATRON_COLOR=  0X008200 # index 4 age = 4 year
DELTATRON_COLOR=  0X005A00 # index 5 age = 5 year

# Output config
_view_growth_types: bool  # make montercarlo growth images
growth_type_window: Print_window_t
phase0g_growth_color: int
phase1g_growth_color: int
phase2g_growth_color: int
phase3g_growth_color: int
phase4g_growth_color: int
phase5g_growth_color: int
_view_deltatron_aging: bool  # make deltatron images during sim
self.deltatron_color_count = 0
deltatron_color: []
deltatron_aging_window: Print_window_t
