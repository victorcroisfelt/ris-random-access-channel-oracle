# Random Access Protocol with Channel Oracle Enabled by a Reconfigurable Intelligent Surface
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

Croisfelt, V., Saggese, F., Leyva-Mayorga, I., Kotaba, R., Gradoni, G., and Popovski, P., “Random Access Protocol with Channel Oracle Enabled by a Reconfigurable Intelligent Surface”, arXiv e-prints, 2022. doi:10.48550/arXiv.2210.04230.

A pre-print version is available on: https://arxiv.org/abs/2210.04230, which has a different content from the published one.

A conference version of this paper is also available on: https://ieeexplore.ieee.org/abstract/document/9833984.

I hope this content helps in your research and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, I also motivate you to share your published results to the public.

If you have any questions and if you have encountered any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
The widespread adoption of Reconfigurable Intelligent Surfaces (RISs) in future practical wireless systems is critically dependent on the integration of the RIS into higher-layer protocols beyond the physical (PHY) one, an issue that has received minimal attention in the research literature. In light of this, we consider a classical random access (RA) problem, where uncoordinated users' equipment (UEs) transmit sporadically to an access point (AP). Differently from previous works, we ponder how a RIS can be integrated into the design of new medium access control (MAC) layer protocols to solve such a problem. We consider that the AP is able to control a RIS to change how its reflective elements are configured, namely, the RIS configurations. Thus, the RIS can be opportunistically controlled to favor the transmission of some of the UEs without the need to explicitly perform channel estimation (CHEST). We embrace this observation and propose a RIS-assisted RA protocol comprised of two modules: Channel Oracle and Access. During channel oracle, the UEs learn how the RIS configurations affect their channel conditions. During the access, the UEs tailor their access policies using the channel oracle knowledge. Our proposed RIS-assisted protocol is able to increase the expected throughput by approximately 60\% in comparison to the slotted ALOHA (S-ALOHA) protocol.

## Content
The code provided here can be used to simulate Figs. 2 to 7 of the paper. The code is organized in the following way:
  - sim_figureX.py: simulation scripts that store the data points need to plot each figure as a .npz file.
  - /data: here you can find the .npz files output by each simulation script. NOTE: you should run the simulation scripts by yourself since the files are too large to share using GitHub.
  - /figs: where the .pdfs of the figures are saved.
  - /plots: here you can find a script named plot_figureX.py for each figure; these scripts load the .npz files saved in /data and output the figures as both pdf and tikz.
  - /src: define classes and functions needed to run the simulations.
  - /tikz: where the .tex's of the figures are saved.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider citing our aforementioned work.

## Acknowledgement
This work was supported by the Villum Investigator Grant “WATER” from the Villum Fonden, Denmark.

```bibtex
@INPROCEEDINGS{9833984,
  author={Croisfelt, Victor and Saggese, Fabio and Leyva-Mayorga, Israel, and Kotaba, Radosław and Gradoni, Gabriele and Popovski, Petar},
  booktitle={2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless Communication (SPAWC)}, 
  title={A Random Access Protocol for RIS-Aided Wireless Communications}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Wireless communication;Access protocols;Reconfigurable intelligent surfaces;Signal processing;Throughput;Reflection;Reconfigurable intelligent surface (RIS);random access},
  doi={10.1109/SPAWC51304.2022.9833984}
}


