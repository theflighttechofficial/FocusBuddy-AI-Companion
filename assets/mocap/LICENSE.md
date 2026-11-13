# Motion-capture library

`library.npz` / `library.json` hold short clips cut from the **CMU Graphics Lab Motion Capture
Database** (http://mocap.cs.cmu.edu), using the MotionBuilder-friendly BVH conversion by Bruce Hahne
(2010 release). `tools/mocap/build_library.py` trims them, turns them to face forward, removes the
travel, makes cycles loop and stores them independent of any skeleton; `focusbuddy/companion/mocap.py`
retargets them onto each model when it loads. `tools/mocap/clips.py` lists every clip and the CMU
trial it comes from.

## Terms

CMU places no restrictions on the use of the original dataset, and the BVH conversion adds none.
From mocap.cs.cmu.edu:

> Use this data! This data is free for use in research and commercial projects worldwide.

## Acknowledgement

The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.
