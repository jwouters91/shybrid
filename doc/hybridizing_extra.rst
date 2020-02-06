Extra features
==============

Exporting template
------------------
A template can be exported as a CSV-file. Every channel is exported as a row in the CSV dump. The order in which the channels are exported is depending on the order of the channels in the probe file. For proper reconstruction, the channels in the probe file / recording are should be ideally ordered based on the actual geometry. More concretely, channels are assumed to be ordered by increasing x- and increasing y-coordinates, with the x-coordinate being the fastest changing variable.

A subset of channels can be exported by using the zoom functionality. All channels which have their leftmost plotted point in the zoom window are considered for exporting.

Importing template
------------------
Ground truth data can also be generated in the absence of initial spike sorting results for a certain recording. This can be obtained by importing an external template in CSV-format, where every row in the CSV-file represents the waveform on a channel. The window size is automatically determined. The horizontal reach parameter will control the width (i.e., the number of channels in the x-direction) of the template on the probe. The offset parameter allows more control about which channel is used as a starting point. The template can also be shaped by adding additional zero-channels to the CSV.

When working with an imported template, the inspect template fit feature will be disabled until a spatial location is chosen and the template is actually inserted in the recording.

Making figures
--------------
The content of the visualization canvas can be saved to an image at any time by pushing the *export plot* button. Remember that the plot control buttons can be used to manipulate the visualization canvas.
