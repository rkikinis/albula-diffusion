// THE INTERPRETER'S FINGERPRINT MATCHES ITS SOURCES (interpreter-code.generated.ts, made by the workspace's
// Contents/tools/extensions.ts fingerprint, which the rebuild runs). A working copy records it; if it lagged behind the
// code, copies read by the old rules would still be used.
import { assertEquals } from "jsr:@std/assert@1";
import { interpreterFingerprint } from "albula/testing";
import { INTERPRETER_CODE } from "./interpreter-code.generated.ts";

Deno.test("the DICOM interpreter's fingerprint matches its sources (run: Contents/tools/extensions.ts fingerprint)", async () => {
  const root = new URL(".", import.meta.url).pathname;
  assertEquals(await interpreterFingerprint(root, "diffusion-vendors.ts", ["interpreter-code.generated.ts"]), INTERPRETER_CODE);
});
