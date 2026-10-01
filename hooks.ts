// WHAT THE DIFFUSION EXTENSION ADDS TO READING DATA, registered through the SDK. Loaded wherever data is read: the app's
// page (with the module), the command-line tools (bids-to-dicom, the copy sweep, the case-library run), and the tests.
//   - DICOM: the b-value and gradient direction of every image, from the standard attributes or a vendor's private
//     ones (diffusion-vendors.ts), as what separates one volume of a series from the next.
//   - BIDS: a session's dwi/ folder, imported as Enhanced MR diffusion objects (bids-dwi.ts).
import { registerBidsKind, registerVolumeInterpreter } from "albula";
import { diffusionInterpreter } from "./diffusion-vendors.ts";
import { bidsDiffusion } from "./bids-dwi.ts";

registerVolumeInterpreter(diffusionInterpreter);
registerBidsKind(bidsDiffusion);
