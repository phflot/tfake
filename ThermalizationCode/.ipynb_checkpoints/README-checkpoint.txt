This directory contains code for the thermalization in the T-FAKE paper.

The main file is "thermalization.py". We used Python 3.10.13.
The Jupyter Notebook display training progress of our model.
In particular, it shows the (batchwise) training loss and some "thermal" T-FAKE images.
We cannot show SEJONG images due to privacy reasons that the
SEJONG authors communicated to us.

The SEJONG dataset cannot be downloaded freely, but one can simply ask the authors.
For more information: https://github.com/usmancheema89/SejongFaceDatabase

The FAKE dataset is freely available via https://github.com/microsoft/FaceSynthetics

The dataset classes are provided and the code should run given paths to the datasets. We omitted all personal paths because of the peer review.

Upon acceptance, the ultimate goal is to provide our landmarker as a tool for practitioners.
