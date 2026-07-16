# GRPO6 Factorized MIL

This variant decomposes each existing pair-class label into two image-level
component-presence targets. A shared component head is trained by MIL over all
DINOSAUR slots; there is no slot assignment, bbox, mask, or slothead target.

The policy receives reward when adding a slot improves the best assignment of
the two true components to two distinct selected slots. This structural bias
tests whether weak label factorization can recover evidence without Oracle
localization supervision. Attention novelty remains only in the stop rule.
