## Reference
HALCON examples
```
HALCON-20.11-Progress\examples\hdevelop\Morphology\Gray-Values\pcb_inspection.hdev

* detect defects ...
gray_opening_shape (Image, ImageOpening, 7, 7, 'octagon')
gray_closing_shape (Image, ImageClosing, 7, 7, 'octagon')
dyn_threshold (ImageOpening, ImageClosing, RegionDynThresh, 75, 'not_equal')

HALCON-20.11-Progress\examples\hdevelop\Applications\Measuring-2D\measure_circuit_width_lines_gauss.hdev

* Extract the tracks
decompose3 (Image, ImageR, ImageG, ImageB)
reduce_domain (ImageG, PcbPart, ImageReduced)
threshold (ImageReduced, Region, 90, 255)
dilation_circle (Region, RegionDilation, 3.5)
opening_rectangle1 (Region, RegionOpening, 8, 8)
dilation_circle (RegionOpening, RegionDilation1, 3.5)
difference (RegionDilation, RegionDilation1, RegionDifference)
connection (RegionDifference, ConnectedRegions)
select_shape (ConnectedRegions, RegionSelected, 'area', 'and', 260, 4595)
union1 (RegionSelected, RegionTracks)
reduce_domain (ImageReduced, RegionTracks, ImageReducedTracks)
* 
* Measure the position and the width of the tracks
lines_gauss (ImageReducedTracks, Lines, 1.5, 1, 8, 'light', 'true', 'bar-shaped', 'true')
select_shape_xld (Lines, SelectedXLD, 'contlength', 'and', 20, 99999)
```

## Dataset

- `references/matching/data/T0517` PCB 深蓝电路
