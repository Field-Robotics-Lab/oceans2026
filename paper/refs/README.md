# Local reference copies

PDFs are named `<bibkey>.pdf`, matching the keys in `../references.bib`. The filename is what creates the link: with `\draftlinkstrue` set in `paper.tex`, every `\cite` whose key has a file here gets a small blue superscript bullet that opens the PDF. Keys with no file here get no marker, so the markers double as an inventory of what we hold.

Everything here came from the publisher's own open-access copy, an author's institutional repository, or arXiv.

**Status: 17 of 31 held.**

## Held (17)

| Bibkey | Authors | Title | Note |
|---|---|---|---|
| `alexandersson2022system` | Alexandersson, Mao, Ringsberg | System Identification of Vessel Manoeuvring Models | Chalmers accepted version |
| `astrom1976identification` | Åström, Källström | Identification of Ship Steering Dynamics | Lund LUP scan |
| `box1976science` | George E. P. Box | Science and Statistics | JASA 1976; earliest citable form of the over-elaboration argument |
| `eriksen2017modeling` | Eriksen, Breivik | Modeling, Identification and Control of High-Speed ASVs | |
| `fossen2011handbook` | Fossen | Handbook of Marine Craft Hydrodynamics and Motion Control | First edition (2011) |
| `morel2023modelling` | Morel, Orihuela, Bejarano | Modelling and Identification of an Autonomous Surface Vehicle | MARTECH 2023; Yellowfish ASV, tabulated parameters |
| `morel2025practical` | Morel, Orihuela, Combastel, Bejarano | Practical Identification Approach for the Actuation Dynamics of ASVs with Minimal Instrumentation | arXiv:2410.00631 extended version |
| `muske2008identification` | Muske, Ashrafiuon, Haas, McCloskey, Flynn | Identification of a Control Oriented Nonlinear Dynamic USV Model | ACC 2008 |
| `sarda2016nmpc` | Sarda, Qu, Bertaska, von Ellenrieder | Station-Keeping Control of an Unmanned Surface Vehicle | arXiv:1702.04941 |
| `sonnenburg2010control` | Sonnenburg, Gadre, Horner, Kragelund, Marcus, Stilwell, Woolsey | Control-Oriented Planar Motion Modeling of Unmanned Surface Vehicles | Extended VT report version |
| `sonnenburg2013modeling` | Sonnenburg, Woolsey | Modeling, Identification, and Control of an Unmanned Surface Vehicle | **Substitution:** this is Sonnenburg's 2013 PhD dissertation, not the JFR article. The dissertation is a superset; Ch. 4–5 carry the model hierarchy and the two-stage identification method. Cite the JFR paper; read this. |
| `suarez2026regularized` | Suárez, Berndt, Abdel-Maksoud | Regularized Machine Learning for System Identification of Ship Free-Running Manoeuvres | arXiv:2606.17121, preprint under review |
| `tanveer2022yaw` | Tanveer, Ahmad | Unmanned Surface Vehicle: Yaw Modeling and Identification | ICEANS 2022 |
| `wirtensohn2013modelling` | Wirtensohn, Reuter, Blaich, Schuster, Hamburger | Modelling and Identification of a Twin Hull-Based Autonomous Surface Craft | MMAR 2013 |
| `xu2022pinn` | Xu, Han, Cheng, Cheng, Ge | A Physics-Informed Neural Network for the Prediction of USV Dynamics | MDPI open access |
| `xu2025review` | Xu, Guedes Soares | Review of System Identification for Manoeuvring Modelling of Marine Surface Ships | |
| `zhang2024gbm` | Zhang, Li, Xiong, He | GBM-ILM: Grey-Box Modeling Based on Incremental Learning and Mechanism for USVs | MDPI open access |

Also present: `fossen94guidance.pdf` (Fossen, *Guidance and Control of Ocean Vehicles*, 1994). Redundant with the 2011 handbook, which is what the paper cites, so it has no bibkey and carries no marker. Delete whenever.

## Not held (14)

### Worth chasing — the argument leans on these

| Bibkey | Authors | Title | Venue | Locator |
|---|---|---|---|---|
| `woo2018dynamic` | Joohyun Woo, Jongyoung Park, Chanwoo Yu, Nakwan Kim | Dynamic Model Identification of Unmanned Surface Vehicles Using Deep Learning Network | Applied Ocean Research 78, 123–133 (2018) | 10.1016/j.apor.2018.06.011 |
| `yasukawa2015mmg` | Hironori Yasukawa, Yasuo Yoshimura | Introduction of MMG Standard Method for Ship Maneuvering Predictions | J. Marine Science and Technology 20(1), 37–52 (2015) | 10.1007/s00773-014-0293-y — **open access**, Springer redirect is the only obstacle |
| `kallstrom1981experiences` | Claes G. Källström, Karl Johan Åström | Experiences of System Identification Applied to Ship Steering | Automatica 17(1), 187–198 (1981) | 10.1016/0005-1098(81)90094-7 — try Lund LUP, the 1976 companion is free there |
| `hann2010simplified` | Christopher E. Hann, Harsha Sirisena, Napasool Wongvanich | Simplified Modeling Approach to System Identification of Nonlinear Boat Dynamics | American Control Conference 2010, 5218–5223 | 10.1109/ACC.2010.5530459 |
| `skelton1989model` | Robert E. Skelton | Model Error Concepts in Control Design | International Journal of Control 49(5), 1725–1753 (1989) | 10.1080/00207178908559735 — carries the "appropriate, not small" criterion the paper builds on |

`woo2018dynamic` is the priority. Gap 2 in the paper asserts they report one-step prediction only; that must be verified before submission.

### Lower priority

| Bibkey | Authors | Title | Venue | Locator |
|---|---|---|---|---|
| `xue2022probabilistic` | Yifan Xue, Xingyao Wang, Hongde Qin, Zhongchao Deng | Probabilistic Identification of Unmanned Surface Vehicles Using Efficient Gaussian Processes with Uncertainty Propagation | IEEE ICUS 2022, 1004–1009 | 10.1109/ICUS55513.2022.9986638 |
| `gevers2005identification` | Michel Gevers | Identification for Control: From the Early Achievements to the Revival of Experiment Design | European Journal of Control 11(4–5), 335–352 (2005) | 10.3166/ejc.11.335-352 |
| `liu2016survey` | Zhixiang Liu, Youmin Zhang, Xiang Yu, Chi Yuan | Unmanned Surface Vehicles: An Overview of Developments and Challenges | Annual Reviews in Control 41, 71–93 (2016) | 10.1016/j.arcontrol.2016.04.018 |
| `wang2020modeling` | Ning Wang et al. (author list unverified) | Modeling and Identification of an Unmanned Surface Vehicle Based on Sea Trials Data | Chinese Automation Congress 2020 | DOI does not resolve — **verify or drop** |

### Don't chase

| Bibkey | Authors | Title | Why not |
|---|---|---|---|
| `bingham2019gazebo` | Bingham, Agüero, McCarrin, Klamo, Malia, Allen, Lum, Rawson, Waqar | Toward Maritime Robotic Simulation in Gazebo | Your own paper |
| `fossen` / `ljung1999system` | Lennart Ljung | System Identification: Theory for the User, 2nd ed. | Book |
| `box1987empirical` | George E. P. Box, Norman R. Draper | Empirical Model Building and Response Surfaces | Book |
| `nomoto1957steering` | Kensaku Nomoto, T. Taguchi, K. Honda, S. Hirano | On the Steering Qualities of Ships | Int. Shipbuilding Progress 4(35), 354–370 (1957). No digital copy in circulation |
| `abkowitz1964lectures` | Martin A. Abkowitz | Lectures on Ship Hydrodynamics: Steering and Manoeuvrability | Report Hy-5, Lyngby (1964). No digital copy in circulation |

To add one: drop the PDF here named `<bibkey>.pdf` and rebuild. Nothing else to edit.
