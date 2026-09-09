# Sampling & Labelling Methodology Note: @AppleSupport Golden Evaluation Benchmark

## 1. Provenance & Dataset Extraction
The evaluation dataset is extracted from the canonical Kaggle dataset **`thoughtvector/customer-support-on-twitter`** (~3M tweets from top brands). 
We focused on **`@AppleSupport`**, one of the most operationally demanding brands in the dataset, characterized by:
- Strict technical isolation protocols (device model, OS version).
- High-stakes privacy/security boundaries (Apple ID lock, 2-Factor Authentication, billing).
- Physical-to-digital boundaries (Genius Bar hardware appointments vs. self-service software resets).
- Real Twitter constraints (140/280 character limits, public vs DM privacy transitions, noise, informal grammar).

## 2. Sampling Strategy
To avoid standard random sampling bias (where over 70% of tweets are generic iOS update complaints), we utilized a **stratified sampling methodology** across 6 explicit operational intents:

| Operational Intent | Description | Golden Set Count | Stratified % |
|---|---|---|---|
| `SOFTWARE_OS_BUG` | iOS/macOS update glitches, battery drain, keyboard lag, audio bugs | 90 | 45.0% |
| `HARDWARE_BATTERY` | Physical battery swelling, battery degradation, screen cracks, ports | 38 | 19.0% |
| `ACCOUNT_SECURITY_ICLOUD` | Locked Apple ID, 2FA failure, unauthorized billing, iCloud lock | 24 | 12.0% |
| `DEVICE_SETUP_FEATURE` | How-to guidance (3D Touch app deletion, AirPods pairing, Family Sharing) | 20 | 10.0% |
| `GENERAL_FEEDBACK_RANT` | Product complaints (headphone jack removal, pricing, snarky venting) | 16 | 8.0% |
| `STORE_REPAIR_ORDER` | Genius Bar booking, repair status tracking, trade-in kit delivery | 12 | 6.0% |
| **Total** | | **200** | **100.0%** |

## 3. Ground Truth Labelling Rules
Each example was hand-verified and annotated according to strict operational criteria:

1. **Intent Taxonomy**:
   - Each customer message is categorized into mutually exclusive primary operational intents based on customer root need, not just superficial keywords.
2. **Escalation Policy Decision (`gold_escalate: true/false`)**:
   - **MUST ESCALATE (`true`)**:
     * **Security / PII**: Any account password reset, locked Apple ID, 2FA code issue, or billing inquiry. Rationale: Apple Support strictly forbids collecting customer credentials or credit card numbers in public Twitter posts.
     * **Physical Hardware / Safety**: Swollen batteries, cracked screens, liquid damage requiring hands-on hardware inspection or Genius Bar diagnostics.
     * **Logistics / Store Dispatch**: Checking personalized store reservation slots or shipping order status.
     * **Exhausted Basic Isolation**: Complex multi-turn software glitches where initial restarts and resets failed to resolve the issue.
   - **CAN AUTO-HANDLE (`false`)**:
     * Feature education, public knowledge base references (e.g. HT201269), standard troubleshooting isolation questions (device model and iOS version check), and general brand feedback acknowledgement.
3. **Human Quality Score (1.0 to 5.0)**:
   - Ground truth human rating evaluating the quality of Apple's historical reply on a standard 5-point rubric:
     * 5: Exceptional brand voice, precise diagnostic isolation question, and accurate link/policy handoff.
     * 4: Strong standard support response with clear next action.
     * 3: Acceptable but generic or slightly blunt canned response.
     * 2: Incomplete resolution or confusing instruction.
     * 1: Inappropriate response, failure to address customer grievance, or policy violation.

## 4. Quality Control & Calibration
- Annotator guidelines were strictly calibrated against Apple Support's real public customer service handbook standards.
- Ground truth escalation rules prioritize **zero false negatives** on privacy/security: A customer with a compromised account or locked Apple ID must NEVER be falsely marked as `auto-handle`.
