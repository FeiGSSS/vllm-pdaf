# PAP: Prefill-Attention-Projection Disaggregation for LLM Serving

> Manuscript status: scaffold only. No paper claims have been approved.

## Abstract

To be written after the core claims and evidence are established.

## 1. Introduction

### 1.1 Problem

To be established.

### 1.2 Key insight

To be established.

### 1.3 Contributions

To be populated only from paper-ready entries in `claims.md`.

## 2. Background and Motivation

### 2.1 LLM serving and KV-cache ownership

To be established with primary-source citations.

### 2.2 Limits of existing deployment architectures

To be established with primary-source citations and controlled evidence.

### 2.3 Motivation for PAP

Prior attention-disaggregation systems already demonstrate partial Decode
Attention offload and load-aware admission, while recent analytical work
models the slowest-worker barrier in \(rA\)-to-\(1F\) deployments
[liang2025adrenaline; song2026analytical]. Barrier-aware KV-load placement is
also prior art when requests remain sticky on Decode workers
[chen2026universal].

PAP's working research question is therefore not whether Attention can be
disaggregated. It is whether full PA-side ownership of Prefill, Decode
Attention, and KV state can support a stateless, highly batched Projection
tier while using efficient KV migration at natural post-Prefill boundaries to
repair multi-round placement. This position remains a hypothesis: current
evidence has not established that migration benefit exceeds its cost or that
PAP has a repeat-stable advantage over tuned PD and fused deployment.

## 3. Design

### 3.1 Architecture

To be synthesized from the canonical PAP design documentation.

### 3.2 Scheduling and placement

To be established.

### 3.3 KV-cache lifecycle and migration

To be established.

### 3.4 Communication and synchronization

To be established.

## 4. Implementation

To be synthesized from the validated implementation boundary.

## 5. Evaluation

### 5.1 Methodology

To be linked to normalized experiment records.

### 5.2 End-to-end performance

No paper-ready result is registered.

### 5.3 Causal analysis and ablations

No paper-ready result is registered.

### 5.4 Scalability and generality

No paper-ready result is registered.

### 5.5 Sensitivity and limitations

No paper-ready result is registered.

## 6. Related Work

To be synthesized from `related-work.md` and `references.bib`.

## 7. Discussion and Limitations

To be updated whenever a claim's scope or counterevidence changes.

## 8. Conclusion

To be written after the paper-level completion audit.
