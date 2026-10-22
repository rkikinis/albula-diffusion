// THE COLOR OF EACH NAMED TRACT -- a numbered version (Ron, 2026-09-25: "modular … and also versioned"); a tract keeps
// its color from patient to patient, and left and right share it. The name card's swatch is this color (Ron,
// 2026-09-30). v1: hues spread by the golden angle over TractCloud's 42 tracts, so neighbors in the list differ; one
// saturation and lightness, readable on the dark 3D background.
import type { RGBA } from "albula";

export const TRACT_COLORS_VERSION = 1;
export const UNNAMED: RGBA = [0.62, 0.62, 0.62, 1];

function hsl(h: number, s: number, l: number): RGBA {
  const a = s * Math.min(l, 1 - l), f = (n: number) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4), 1];
}

/** The color of tract `index` (TractCloud's order, model.json "tracts"). The last one, Other, is gray. */
export function tractColor(index: number, count: number): RGBA {
  if (index < 0 || index >= count - 1) return UNNAMED;
  return hsl((index * 137.508) % 360, 0.72, 0.58);
}
