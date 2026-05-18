---
title:          "S+t-SNE - Bringing Dimensionality Reduction to Data Streams"
date:           2024-04-16
selected:       false
pub:            "Advances in Intelligent Data Analysis XXII"
pub_date:       "2024"
abstract: >-
  We adapt t-SNE to data streaming scenarios. We solve the issues of massive data accumulation by constructing convex hulls over the embeddings. To handle drift we develop exponential cobweb slicing to track the regions of the hull that grew obsolute as more data is incorporated in the projection. We validate or findings on synthetic data and on MNIST.
cover:          /assets/images/covers/moving-hull.gif
authors:
- Pedro C. Vieira*#
- João P. Montrezol*
- João T. Vieira
- João Gama
links:
  Paper: https://link.springer.com/chapter/10.1007/978-3-031-58553-1_8
  Code: https://github.com/pedrv/s--t-sne
---