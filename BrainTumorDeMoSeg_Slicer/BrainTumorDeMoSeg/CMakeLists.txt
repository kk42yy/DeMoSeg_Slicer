cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BrainTumorDeMoSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BrainTumorDeMoSeg")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Kaixiang Yang (Huazhong University of Science and Technology)")
set(EXTENSION_DESCRIPTION "This is an automatic segmenting tools for  incomplete brain tumors segmentation. However, 4-modality images should be  skull-stripped and co-registered.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/BrainTumorDeMoSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/BrainTumorDeMoSeg/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BrainTumorDeMoSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
