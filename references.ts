// THE PAPERS THE DIFFUSION CODE IS BUILT ON -- one list, shown in the Diffusion module's Help & Acknowledgment (Ron,
// 2026-09-29: "We need the reference for what papers we used in the help message"). Each entry says what in the code
// rests on it. Versioned: an entry is added, never silently rewritten. Verified = the citation matches the publisher's record
// (Crossref, by DOI; Semantic Scholar where there is no DOI) or was read at its source
// during the work (2026-09-28/29); unverified entries are cited from memory and marked so, to check before release.

// v2 (2026-09-30): the two WMQL papers left with the WMQL code (Ron: "delete the wmql code"). v3: TractCloud and its atlas.
// v4 (2026-10-01, critic finding 11): dcm2niix's own citation; every entry checked against Crossref (Semantic Scholar
// for the CDMRI workshop paper, which has no DOI) -- Baumgartner et al. 2012's author list corrected, TractCloud cited
// by its pages (Crossref gives no volume).
export const REFERENCES_VERSION = 4;

export interface Reference { cite: string; link?: string; usedFor: string; verified: boolean }

export const DIFFUSION_REFERENCES: Reference[] = [
  // How to cite the platform this follows.
  { cite: "Zhang F, Noh T, Juvekar P, Frisken SF, Rigolo L, Norton I, Kapur T, Pujol S, Wells W III, Yarmarkovich A, Kindlmann G, Wassermann D, San Jose Estepar R, Rathi Y, Kikinis R, Johnson HJ, Westin CF, Pieper S, Golby AJ, O'Donnell LJ. SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization. JCO Clinical Cancer Informatics 4:299-309, 2020.", link: "https://ascopubs.org/doi/full/10.1200/CCI.19.00141", usedFor: "the clinical workflow (two-tensor UKF with free water near tumors) this module follows", verified: true },
  { cite: "Norton I, Ibn Essayed W, Zhang F, Pujol S, Yarmarkovich A, Golby AJ, Kindlmann G, Wassermann D, San Jose Estepar R, Rathi Y, Pieper S, Kikinis R, Johnson HJ, Westin CF, O'Donnell LJ. SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21):e101-e103, 2017.", link: "https://aacrjournals.org/cancerres/article/77/21/e101/662618/", usedFor: "SlicerDMRI, the reference platform", verified: true },
  // UKF tractography.
  { cite: "Malcolm JG, Shenton ME, Rathi Y. Filtered multitensor tractography. IEEE Transactions on Medical Imaging 29(9):1664-1675, 2010.", link: "https://pmc.ncbi.nlm.nih.gov/articles/PMC3045040/", usedFor: "UKF tractography (ukf.ts, ukf-gpu.ts): the two-tensor model and the unscented Kalman filter", verified: true },
  { cite: "Reddy CP, Rathi Y. Joint multi-fiber NODDI parameter estimation and tractography using the unscented information filter. Frontiers in Neuroscience 10:166, 2016.", link: "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2016.00166/full", usedFor: "the information form of the filter (ukf.ts)", verified: true },
  { cite: "Baumgartner C, Michailovich O, Levitt J, Pasternak O, Bouix S, Westin CF, Rathi Y. A unified tractography framework for comparing diffusion models on clinical scans. MICCAI Workshop on Computational Diffusion MRI (CDMRI), Nice, 2012.", usedFor: "free water inside the filter (ukf.ts)", verified: true },
  { cite: "UKFTractography, Brigham and Women's Hospital (github.com/pnlbwh/ukftractography), UKF Tractography Contribution and Software License Agreement.", link: "https://github.com/pnlbwh/ukftractography", usedFor: "the code ukf.ts is ported from", verified: true },
  // The tensor, maps, tracking.
  { cite: "Basser PJ, Mattiello J, LeBihan D. MR diffusion tensor spectroscopy and imaging. Biophysical Journal 66(1):259-267, 1994.", link: "https://doi.org/10.1016/S0006-3495(94)80775-1", usedFor: "the diffusion tensor (tensor.ts)", verified: true },
  { cite: "Salvador R, Peña A, Menon DK, Carpenter TA, Pickard JD, Bullmore ET. Formal characterization and extension of the linearized diffusion tensor model. Human Brain Mapping 24(2):144-155, 2005.", link: "https://doi.org/10.1002/hbm.20076", usedFor: "weighted least squares (tensor.ts)", verified: true },
  { cite: "Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage 81:335-346, 2013.", link: "https://doi.org/10.1016/j.neuroimage.2013.05.028", usedFor: "weighted least squares (tensor.ts)", verified: true },
  { cite: "Pajevic S, Pierpaoli C. Color schemes to represent the orientation of anisotropic tissues from diffusion tensor data: application to white matter fiber tract mapping in the human brain. Magnetic Resonance in Medicine 42(3):526-540, 1999.", link: "https://doi.org/10.1002/(SICI)1522-2594(199909)42:3<526::AID-MRM15>3.0.CO;2-J", usedFor: "color FA (tensor.ts)", verified: true },
  { cite: "Basser PJ, Pajevic S, Pierpaoli C, Duda J, Aldroubi A. In vivo fiber tractography using DT-MRI data. Magnetic Resonance in Medicine 44(4):625-632, 2000.", link: "https://doi.org/10.1002/1522-2594(200010)44:4<625::AID-MRM17>3.0.CO;2-O", usedFor: "single-tensor streamline tracking (tracking.ts)", verified: true },
  { cite: "Otsu N. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics 9(1):62-66, 1979.", link: "https://doi.org/10.1109/TSMC.1979.4310076", usedFor: "the brain mask (tensor.ts brainMask)", verified: true },
  { cite: "Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics 8:8, 2014.", link: "https://doi.org/10.3389/fninf.2014.00008", usedFor: "the independent check of the tensor fit (a check tool, not a dependency)", verified: true },
  // Distortion correction.
  { cite: "Chang H, Fitzpatrick JM. A technique for accurate magnetic resonance imaging in the presence of field inhomogeneities. IEEE Transactions on Medical Imaging 11(3):319-329, 1992.", link: "https://doi.org/10.1109/42.158935", usedFor: "the reversed phase-encoding principle (distortion.ts)", verified: true },
  { cite: "Ruthotto L, Kugel H, Olesch J, Fischer B, Modersitzki J, Burger M, Wolters CH. Diffeomorphic susceptibility artifact correction of diffusion-weighted magnetic resonance images. Physics in Medicine and Biology 57(18):5715-5731, 2012.", link: "https://iopscience.iop.org/article/10.1088/0031-9155/57/18/5715", usedFor: "the variational model with the Jacobian barrier (distortion.ts)", verified: true },
  { cite: "Macdonald J, Ruthotto L. Improved susceptibility artifact correction of echo-planar MRI using the alternating direction method of multipliers. Journal of Mathematical Imaging and Vision, 2018. arXiv:1607.00531.", link: "https://arxiv.org/abs/1607.00531", usedFor: "the full formulation, solver and parameters implemented (distortion.ts)", verified: true },
  // Reading the scan: the second opinion.
  { cite: "Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods 264:47-56, 2016.", link: "https://doi.org/10.1016/j.jneumeth.2016.03.001", usedFor: "dcm2niix, run on every diffusion scan from the database as a second opinion (vendor/dcm2niix/)", verified: true },
  // Naming tracts.
  { cite: "Xue T, Chen Y, Zhang C, Golby AJ, Makris N, Rathi Y, Cai W, Zhang F, O'Donnell LJ. TractCloud: registration-free tractography parcellation with a novel local-global streamline point cloud representation. MICCAI 2023, Lecture Notes in Computer Science, pp. 409-419.", link: "https://doi.org/10.1007/978-3-031-43993-3_40", usedFor: "TractCloud: the network and its trained weights (tractcloud/)", verified: true },
  { cite: "Zhang F, Wu Y, Norton I, Rigolo L, Rathi Y, Makris N, O'Donnell LJ. An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. NeuroImage 179:429-447, 2018.", link: "https://doi.org/10.1016/j.neuroimage.2018.06.027", usedFor: "the atlas of 800 fiber clusters TractCloud's names come from", verified: true },
  // Data.
  { cite: "Aerts H, Schirner M, Jeurissen B, Van Roost D, Achten E, Ritter P, Marinazzo D. Modeling brain dynamics in brain tumor patients using The Virtual Brain. eNeuro 5(3):ENEURO.0083-18.2018, 2018. Data: OpenNeuro ds001226 (CC0).", link: "https://openneuro.org/datasets/ds001226", usedFor: "the development and test cases", verified: true },
];

/** The list as plain text lines for a help panel. */
export function referencesText(): string[] {
  return DIFFUSION_REFERENCES.map((r, i) => `${i + 1}. ${r.cite}${r.link ? ` ${r.link}` : ""} — ${r.usedFor}${r.verified ? "" : " (citation to be checked)"}`);
}
