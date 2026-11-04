# Literature Review: Single-Channel EEG Features for Sleep Staging

> **Data provenance.** This run used a *synthetic* polysomnography surrogate (origin: `synthetic`), not human recordings. Every association reported below is a property of that simulation and must not be read as a statement about human sleep. Re-run with network access to use Sleep-EDF.

## Retrieval status

| Field | Value |
| --- | --- |
| retrieval mode | offline |
| records | 4 |
| live retrieval available | no |
| full texts retrieved | 0 |
| queries planned | 0 |
| queries executed | 0 |
| citation validation | ok |

### What could not be verified
- no network access was detected, so live bibliographic retrieval was skipped
- a previously retrieved bibliography exists at /Users/faraz/agentic-feature-exploration/data/metadata/papers.json and was merged in

This review was produced **without live bibliographic retrieval**. The records below are the anchor sources named in the project brief plus any previously cached bibliography; their metadata has not been checked against the publisher in this run. No statement in this document should be read as a verified literature finding, and no numeric quantity is attributed to any author without a retrieved full text.

## Bibliography

| title | year | venue | identifier | id type | evidence |
| --- | --- | --- | --- | --- | --- |
| AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms | n/a | n/a | https://github.com/emadeldeen24/AttnSleep | url | unverified-offline |
| BIOT: Biosignal Transformer for Cross-data Learning in the Wild | 2023 | NeurIPS 2023 (as stated in the project brief) | https://arxiv.org/abs/2305.10351 | arxiv | unverified-offline |
| Sleep-EDF Expanded (version 1.0.0) | 2013 | PhysioNet | https://physionet.org/content/sleep-edfx/1.0.0/ | url | unverified-offline |
| MNE-Python sleep staging tutorial (clinical tutorial 60) | n/a | MNE-Python documentation | https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html | url | unverified-offline |

## Overview of single-channel EEG sleep staging

Records matched to this topic were retrieved at metadata level only, so no claim from their full text is used here:

- AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms (n.d.; https://github.com/emadeldeen24/AttnSleep)
- MNE-Python sleep staging tutorial (clinical tutorial 60) (n.d.; https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)

## Conventional EEG features

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## N1/N2/REM classification difficulty

Records matched to this topic were retrieved at metadata level only, so no claim from their full text is used here:

- AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms (n.d.; https://github.com/emadeldeen24/AttnSleep)
- MNE-Python sleep staging tutorial (clinical tutorial 60) (n.d.; https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)

## Nonlinear feature engineering

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## Event-based EEG features

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## Feature selection techniques

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## Statistical evaluation methods

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## Deep-learning approaches

Records matched to this topic were retrieved at metadata level only, so no claim from their full text is used here:

- AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms (n.d.; https://github.com/emadeldeen24/AttnSleep)
- BIOT: Biosignal Transformer for Cross-data Learning in the Wild (2023; https://arxiv.org/abs/2305.10351)

## Research gaps

No retrieved source in this run addresses this topic directly. The associated feature families are therefore treated as *literature-inspired but unverified in this run* and are labelled as such in the feature catalog.

## Candidate experimental hypotheses

Records matched to this topic were retrieved at metadata level only, so no claim from their full text is used here:

- AttnSleep: An Efficient Sleep Staging Model Based on Attention Mechanisms (n.d.; https://github.com/emadeldeen24/AttnSleep)

## Research gaps this project targets
- Single-channel montages discard EOG/EMG evidence, so REM is separated from N1/N2 by EEG alone.
- N1 is short, transitional and unbalanced; per-stage metrics are reported rather than accuracy alone.
- Event features (spindles, K-complexes, slow waves) are often conflated with band power; both are computed here and compared.
- Nonlinear and information-theoretic features are rarely validated on synthetic signals with known properties.
- Epoch-level random splits leak subject identity; subject-grouped splits with a saved manifest are used here.

## Candidate experimental hypotheses
- H1: spectral concentration measures separate N2 from N1/REM more strongly than absolute band power.
- H2: within-epoch spectral instability is higher in transitional stages (N1, and wake-sleep transitions).
- H3: event-based features (spindle, slow-wave density) carry information not recoverable from band power alone.
- H4: nonlinear complexity features are stage-dependent even after amplitude normalisation.
- H5: combining spectral, nonlinear and event families improves per-stage balanced accuracy over any single family.
- H6: any predictive advantage of a novel feature family survives subject-grouped held-out evaluation.

## How each hypothesis is tested
H1-H4 are tested with the repeated-measures comparisons in `reports/statistical_analysis.md` (subject x stage means, FDR-corrected). H5 is tested by the ablation study in `reports/final_research_report.md`. H6 is tested on the held-out test subjects and is the only basis on which a feature is described as predictively useful.
