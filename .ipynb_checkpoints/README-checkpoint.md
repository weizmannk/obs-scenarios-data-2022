# LIGO/Virgo/KAGRA Observing Capabilities: Simulated Detections and Localization for O4 and O5 (October 2022 edition)

This repository contains the light files of the Simulated detections and localizations for LIGO, Virgo, and KAGRA.

Il est constituer de la simulation de deux distributions Petrov et Farah

Farah is a distribution drawn from the Power Law+ Dip+ Break model of Farah el al.2022

In Petrov distribution  We follow Petrov et al. 2022, we simulate realistic distributions of compact binary coalescences,
including their mass, spin, and sky locations, as well as their probability of detection based on the
gravitational-wave detector configurations and detection pipeline thresholds.


This repo contains only the light files from the two simulations in order to allow a fast access for the analysis and the treatment of the light curves with nmma. The most important file is: injections.dat , which contains the parameters of masses, spins, distances of coalescences ..... they contain the events which passed the SNR of the detectors. Therefore can be used to generate mass ejecta (for BNS and NSBH populations) for the electromagnetic (EM) counterpart of the gravitational waves in order to test the performance of detections of telescopes in particular ZTF and Rubin (LSST) through photometry. As well as to simulate gravitational waves (GW) using parallel-bilby.

Both GW + EM can be used to estimate the equation of state of EOS neutron stars.
