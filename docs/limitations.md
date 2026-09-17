# Limitations and open questions

- **Synthetic results:** The current benchmark is a deliberately simple, structured fact-acquisition task with a perfect verifier. A strong result here is evidence of functioning plumbing, not of general semantic novelty extraction.
- **No DeepSeek internal integration:** No foundation-model weights are loaded, inspected at runtime, fine-tuned, or modified. The architecture document describes proposed work.
- **API status:** The optional API adapter is mock-tested only. It makes network calls only when explicitly selected. Its output remains a hypothesis.
- **Cheap feature limits:** Claim keys and goals are supplied. Retrieval is exact-key/recency based. Lexical distance and source strings do not establish semantic novelty or evidence independence.
- **Myopic learning:** The heads learn immediate score improvement from a single memory write. They cannot capture synergistic clues, delayed usefulness, or exploration value in general.
- **Imperfect calibration:** Ridge outputs are unconstrained value estimates, not probabilities. The controller can miss valuable events or select worthless ones. Deployment exploration, calibration, and off-policy corrections are not implemented.
- **Simulated compute:** Fixed cost units do not represent actual GPU time, tokens, memory bandwidth, dollars, or offline training costs. No real-world efficiency claim is made.
- **Verification trust:** A registry or investigator defines the trust boundary. The software cannot make incorrect registry entries true. The fixture prevents false verified answers by construction.
- **Simplified memory:** Verified conflicts supersede by arrival order for the same scoped key. Provisional entries never answer queries. Temporal validity and domain-specific evidence aggregation need additional mechanisms.
- **Limited storage management:** Only deferred observations have a capacity limit. Beliefs and audit records can grow without bound. Learned forgetting, replay, and automatic consolidation are not implemented.
- **Operational maturity:** This is an alpha, single-process reference implementation without a hosted service, access-control layer, full migration system, or production serving integration.

The research opportunity is to discover which restrictions can be relaxed while preserving useful selection at low overhead. Contributions should state what they demonstrate and what remains hypothetical.
