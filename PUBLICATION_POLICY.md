# Publication scope

This release includes source, unit tests, dependency names, documentation, and
the optional VAD alignment script required by its test. It excludes datasets,
voice samples, pretrained/trained weights, experiment outputs, raw logs,
machine-specific report scripts, temporary files, installed dependencies,
caches, and credentials. Ignored files must not be force-added.

Training logs and inference sidecars can contain absolute source paths, speaker
identifiers, and per-recording metadata. They are generated locally and should
be reviewed and anonymized before any separate publication.

Only summarized figures/tables with reviewed provenance should be selected for
a future results release. Audio and weights require a separate rights review.

The source release preserves existing remote Git history. Removing a file from
the current tree does not remove it from previous commits. Historical demo
audio and smoke outputs therefore remain accessible in older commits. Local
unpublished commits containing RTVC weights are not ancestors of this release.
