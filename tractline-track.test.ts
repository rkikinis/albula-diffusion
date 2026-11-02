// Tracking rule 6 is rule 3 tracked by tractline (tracking-rules.ts, tractline-track.ts): the same shell, brain and
// thresholds, so the only difference between the two is who follows the fibers.
import { assertEquals } from "jsr:@std/assert@1";
import { TRACKING_RULES } from "./tracking-rules.ts";
import { TRACTLINE } from "./tractline-track.ts";

Deno.test("tracking rule 6 is rule 3's settings, tracked by tractline at one pinned commit", () => {
  const { id: _a, tracker, ...six } = TRACKING_RULES[6];
  const { id: _b, ...three } = TRACKING_RULES[3];
  assertEquals(tracker, "tractline");
  assertEquals(six, three);
  assertEquals(TRACTLINE.commit.length, 40);
});
