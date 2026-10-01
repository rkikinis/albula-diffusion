# dcm2niix, WebAssembly (vendored) — the diffusion reading's second opinion

Chris Rorden's dcm2niix (github.com/rordenlab/dcm2niix), as the WebAssembly build its own repository publishes
(`js/` folder) on npm as **`@niivue/dcm2niix` 1.3.20260724** — the same dcm2niix v1.0.20260724 the tests compare
against. Copied here at that pinned version and never fetched at run time. Plain build (no JPEG-LS / JPEG 2000 codecs):
BSD-style license (`LICENSE.txt`, dcm2niix's own), no GPL. `dcm2niix.wasm` SHA-256 begins 4bc755ab7fcea339.

Why it is here (Ron, 2026-09-30: "2 yes"; Lauren O'Donnell's suggestion, `Contents/docs/decision-dcm2niix-inside-2026-09-29.md`
in the workspace): Albula reads diffusion DICOM with its own reader, which keeps the DICOM record; dcm2niix, actively
maintained and first to learn new scanner tags, reads the same files as a SECOND OPINION on every diffusion scan, and
a disagreement in b-values or directions is said before anyone makes tracts (`../../second-opinion.ts`).

Files as published: `index.js` (the `Dcm2niix` class; starts `worker.js` beside itself), `worker.js`, `dcm2niix.js`
(Emscripten glue), `dcm2niix.wasm`. The rebuild copies them to `webgpu/vendor/dcm2niix/` beside the app.

To update: download the tarball from the npm registry (`https://registry.npmjs.org/@niivue/dcm2niix/-/dcm2niix-<v>.tgz`),
copy `dist/{index,worker,dcm2niix}.js` and `dist/dcm2niix.wasm`, change the version here and in the workspace's
`DEPENDING-ON-OTHER-PEOPLE.md`, rebuild, and load a diffusion scan from each vendor set: the second opinion must still
agree.
