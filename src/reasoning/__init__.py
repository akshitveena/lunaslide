"""Stage 3 — reasoning over Stage 1 perception and Stage 2 physics.

Consumes the visual evidence (boulders, craters, shadow, debris) and the
physical hazard (slope failure and its sensitivity to descent-engine vibration)
for a site, weighs the conflicting signals, and returns a landing verdict with
its reasons.  The core rule is epistemic: a site can never be cleared GO while
evidence is missing (an untrained detector, or terrain hidden in shadow).
"""
