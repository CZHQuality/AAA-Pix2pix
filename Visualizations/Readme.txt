
We provide 3 visualizations (GIF format) for your reference.

Each GIF contains two parts:

Part-I: In the begining still frames: 
the upper-left region is the original clean image, the bottom-left region is the ground-truth output of the clean image, the upper-right region is the guide image, the bottom-right region is the ground-truth output of the guide image

Part-II: In the following dynamic frames: 
the upper-left region is the crafted adversarial example, the upper-right region is the normalized perturbation (obtained by elemen-wise subtraction of clean image and adversarial example, and normalized by min-max normalization for better obvervation).
The bottom regions are the outputs of two black-box target models on the crafted adversarial example.
The timestamp denotes the iterations.
