"""
Download the real anatomical surfaces this app can optionally use.

WHAT IS DOWNLOADED
------------------
FreeSurfer's `fsaverage` template from TemplateFlow:

  * pial surfaces (left/right)  - the real folded cortical surface, averaged
    across 40 subjects. This is why the gyri look like gyri.
  * sulc                        - sulcal depth per vertex, used for shading.
  * Desikan-Killiany atlas      - a real anatomical parcellation, so region
    boundaries follow anatomy instead of my geometric guesses.

LICENSING / ATTRIBUTION
-----------------------
fsaverage is distributed as part of FreeSurfer and is subject to the
FreeSurfer licence terms. It is downloaded at first run and cached locally;
it is deliberately NOT committed to this repository, so redistribution is
not our problem and the user obtains it from the original source.

  fsaverage:  Fischl et al. (1999), Hum. Brain Mapp. 8(4):272-284
  Desikan-Killiany: Desikan et al. (2006), NeuroImage 31(3):968-980
  TemplateFlow: templateflow.org

HONESTY NOTE
------------
fsaverage is an *average of many brains*, not any individual's brain, and
certainly not the user's. Using it makes the anatomy real; it does not make
the simulation drawn on top of it a measurement of anyone.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import Dict

BASE = "https://templateflow.s3.amazonaws.com/tpl-fsaverage/"

# den-41k is fsaverage6: 40 962 vertices per hemisphere. That is the sweet
# spot - enough to resolve individual gyri, light enough to render at 60fps.
# den-10k is available as a low-detail fallback.
FILES: Dict[str, str] = {
    "pial_L":  "tpl-fsaverage_hemi-L_den-41k_pial.surf.gii",
    "pial_R":  "tpl-fsaverage_hemi-R_den-41k_pial.surf.gii",
    "sulc_L":  "tpl-fsaverage_hemi-L_den-41k_sulc.shape.gii",
    "sulc_R":  "tpl-fsaverage_hemi-R_den-41k_sulc.shape.gii",
    "aparc_L": "tpl-fsaverage_hemi-L_den-41k_atlas-Desikan2006_seg-aparc_dseg.label.gii",
    "aparc_R": "tpl-fsaverage_hemi-R_den-41k_atlas-Desikan2006_seg-aparc_dseg.label.gii",
    # Destrieux gives 75 parcels naming individual gyri AND sulci, which is
    # what makes "what am I looking at?" answerable at anatomical precision.
    "destrieux_L": "tpl-fsaverage_hemi-L_den-41k_atlas-Destrieux2009_dseg.label.gii",
    "destrieux_R": "tpl-fsaverage_hemi-R_den-41k_atlas-Destrieux2009_dseg.label.gii",
    # Yeo 2011 functional networks. This is what makes it possible to show
    # the default mode network honestly - as a distributed network derived
    # from resting-state data in 1000 subjects, rather than a region we drew.
    "yeo_L": "tpl-fsaverage_hemi-L_den-41k_atlas-Yeo2011_seg-7n_dseg.label.gii",
    "yeo_R": "tpl-fsaverage_hemi-R_den-41k_atlas-Yeo2011_seg-7n_dseg.label.gii",
}

# The cortical surface files above are all fsaverage. Subcortical structures
# are not surfaces at all - they are labelled volumes - so they come from a
# different template, and the two cannot share a base URL.
#
# This is the file that lets the app stop drawing subcortical structures as
# hand-shaped blobs. `aseg` is FreeSurfer's automatic subcortical
# segmentation: thalamus, caudate, putamen, pallidum, hippocampus, amygdala,
# accumbens, ventricles, brainstem and cerebellum, each as a voxel label.
VOL_BASE = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"

VOL_FILES: Dict[str, str] = {
    # res-02 is 2 mm, ~1.1 M voxels, 86 KB. res-01 would be eight times the
    # work for detail finer than the segmentation itself is reliable at.
    "aseg": "tpl-MNI152NLin2009cAsym_res-02_seg-aseg_dseg.nii.gz",
    # Tissue probability for white matter. Used to route the pathway curves
    # through white matter instead of straight through ventricles and cortex.
    # Not diffusion data and not tractography - see tracts.py.
    "wm": "tpl-MNI152NLin2009cAsym_res-02_label-WM_probseg.nii.gz",
}

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_assets", "fsaverage")

VOL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_assets", "mni152")

NOTICE = (
    "Cortical surface: FreeSurfer fsaverage template (Fischl et al., 1999),\n"
    "parcellated with the Desikan-Killiany atlas (Desikan et al., 2006) and\n"
    "the Destrieux atlas (Destrieux et al., 2010), from TemplateFlow.\n"
    "Subcortical structures: FreeSurfer aseg segmentation of the\n"
    "MNI152NLin2009cAsym template at 2 mm, also from TemplateFlow.\n"
    "An average of 40 brains - not yours."
)


def paths() -> Dict[str, str]:
    out = {k: os.path.join(ASSET_DIR, v) for k, v in FILES.items()}
    out.update({k: os.path.join(VOL_DIR, v) for k, v in VOL_FILES.items()})
    return out


def have_assets() -> bool:
    """True if every required file is present and non-trivial in size."""
    return all(os.path.exists(p) and os.path.getsize(p) > 1024
               for p in paths().values())


def fetch(force: bool = False, quiet: bool = False) -> Dict[str, str]:
    """Download any missing files. Returns the path map."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(VOL_DIR, exist_ok=True)
    out = paths()
    for key, name in list(FILES.items()) + list(VOL_FILES.items()):
        dest = out[key]
        if not force and os.path.exists(dest) and os.path.getsize(dest) > 1024:
            continue
        url = (VOL_BASE if key in VOL_FILES else BASE) + name
        if not quiet:
            print(f"  downloading {name} ...", end="", flush=True)
        tmp = dest + ".part"
        with urllib.request.urlopen(url, timeout=120) as r, \
                open(tmp, "wb") as fh:
            fh.write(r.read())
        os.replace(tmp, dest)
        if not quiet:
            print(f" {os.path.getsize(dest) // 1024} KB")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv
    print("Fetching real anatomical surfaces into:")
    print(f"  {ASSET_DIR}\n")
    try:
        fetch(force=force)
    except Exception as e:                                # noqa: BLE001
        print(f"\nDownload failed: {e}")
        print("The app still runs - it falls back to the procedural cortex.")
        return 1
    print("\n" + NOTICE)
    print("\nDone. Start the app with:  python -m neuroforge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
