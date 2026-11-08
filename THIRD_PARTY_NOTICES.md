# Third-party sources

HeaderShield was developed from the local VSMask research codebase.
The original README and legacy voice conversion/attack code credit
[Defending Your Voice: Adversarial Attack on Voice Conversion](https://arxiv.org/abs/2005.08781)
and its [Attack-VC implementation](https://github.com/cyhuang-tw/attack-vc).
Original provenance and attribution must be retained. This repository does not
assign a new blanket license to third-party code or claim ownership of it.

The vendored Python files under `external/rtvc/` originate from
[Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning).
Its upstream MIT license is reproduced at `external/rtvc/LICENSE`.
These are a local research copy and may differ from the current upstream version.

AIHub/VCTK data, example voice recordings, pretrained weights, and generated
recordings are not distributed in the current source tree. Obtain any required
data and weights from their original providers under their respective terms.
