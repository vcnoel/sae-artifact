# Artifacts in ablation-based causal-effect measurement of SAE latents

Four confounds in the standard way of measuring what a sparse-autoencoder latent
does, each with a mechanism, a control, and the number it moves. Endpoint: the
trained-versus-random gap after controlling all four.

    src/                analysis scripts (run from repo root)
    src/prior_candidates/  scripts from abandoned earlier candidates, kept for provenance
    results/            CSVs and text reports
    data/              SAE checkpoints (gitignored, ~1.7 GB)
    paper/OUTLINE.md   working outline
    PREREG.md          pre-registered predictions and decision rules
    RELEASE_NOTES.md   defects found and what prevents recurrence

Model: gemma-2-2b, residual stream layer 12. Dictionaries: Gemma Scope
16k/L0-82, plus a matched pair trained here (TopK k=82, 12M tokens, one
unconstrained and one soft-frozen at decoder cosine 0.8 to init).

Scripts expect to run from the repo root: `python src/<name>.py`.
