arXiv Submission Package: Volatility-Driven Decay
===================================================

Title:  Volatility-Driven Decay: Adaptive Memory Retention for RAG
        Systems Under Unknown Drift
Author: Abe Diaz, Independent Researcher
Date:   February 2026

Contents
--------

main.tex
    Complete LaTeX source for the paper. Compiles with pdflatex.
    Uses standard packages: amsmath, amssymb, graphicx, hyperref,
    booktabs, longtable, geometry, lmodern, microtype, parskip.

    Sections:
      1. Introduction (subsections 1.1-1.5)
      2. Related Work (subsections 2.1-2.9)
      3. The VDD Method (subsections 3.1-3.6)
      4. Experimental Validation (subsections 4.1-4.11)
      5. Discussion (subsections 5.1-5.5)
      6. Limitations (14 items)
      7. Future Work
      8. Conclusion
      References (53 entries)
      Appendix A: Complete Experimental Results (A.1-A.6)
      Appendix B: Statistical Details (B.1-B.4)
      Appendix C: Integration Guide (C.1-C.3)
      Appendix D: Hyperparameter Selection Guide

figures/
    Experiment result plots from the results/ directory:
      02_simulation_real.png       - VDD error reduction demonstration
      05_ablation_heatmap.png      - Lambda parameter ablation heatmap
      08_baseline_comparison.png   - 7-baseline comparison
      21_effective_lambda.png      - Effective lambda visualization
      29_sigmoid_heatmap.png       - Sigmoid parameter sensitivity
      31_adaptive_baselines.png    - Adaptive baseline comparison
      32_bimodality_tests.png      - Bimodality test results
      33_three_domain.png          - Three-domain validation
      36_real_embedding_suite.png  - Hash vs real embedding comparison
      37_activation_and_baselines.png - Activation function ablation

README.txt
    This file.

Compilation
-----------

    pdflatex main.tex
    pdflatex main.tex   (run twice for cross-references)

No BibTeX step is required; the bibliography is inline using
thebibliography environment.

Note: Figures are included in the figures/ directory but are not
currently referenced via \includegraphics in main.tex. The paper's
data is presented entirely in LaTeX tables. To add figures, use:

    \begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/FILENAME.png}
    \caption{Caption text.}
    \end{figure}

Repository
----------

Full source code, experiments, and dataset:
https://github.com/abe238/volatility-driven-decay

License: Apache-2.0
