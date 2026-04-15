type PhaseDisplayCopy = {
  label: string;
  detail?: string;
  thinking?: string;
};

const DEFAULT_THINKING_TEXT = "Working through your request";

const PHASE_DISPLAY_COPY: Record<string, PhaseDisplayCopy> = {
  approval: {
    label: "Waiting on your approval",
    detail: "Ready to continue when you are.",
  },
  autonomous_cycle: {
    label: "Driving your request forward",
    detail: "Keeping progress moving with focus.",
  },
  calculator_evidence: {
    label: "Validating critical calculations",
    detail: "Checking the numbers that matter.",
  },
  code_verify: {
    label: "Validating with code",
    detail: "Confirming the result against the code.",
  },
  context_hydrate: {
    label: "Loading your working context",
    detail: "Gathering the details needed next.",
  },
  context_policy: {
    label: "Tailoring context to your request",
    detail: "Focusing the work around your goals.",
  },
  critique_round_1: {
    label: "Reviewing the leading options",
    detail: "Comparing the strongest directions carefully.",
  },
  critique_round_2: {
    label: "Refining the best direction",
    detail: "Sharpening the strongest path forward.",
  },
  deliberation: {
    label: "Selecting the strongest approach",
    detail: "Choosing the best path for you.",
  },
  direct_response: {
    label: "Preparing your answer",
    detail: "Moving straight to a clear response.",
  },
  execution_router: {
    label: "Selecting the strongest approach",
    detail: "Choosing the best path for you.",
  },
  expert_council: {
    label: "Reviewing the strongest answer",
    detail: "Applying extra scrutiny where needed.",
  },
  fast_dialogue: {
    label: "Preparing your answer",
    detail: "Moving straight to a clear response.",
  },
  finalize: {
    label: "Preparing your final response",
    detail: "Shaping the answer for clarity.",
  },
  focused_team: {
    label: "Bringing the best thinking together",
    detail: "Combining strengths around your request.",
  },
  intake: {
    label: "Reviewing your request",
    detail: "Aligning on what matters most.",
  },
  iterative_research: {
    label: "Exploring the strongest evidence",
    detail: "Digging into the most relevant support.",
  },
  knowledge: {
    label: "Loading relevant context",
    detail: "Bringing in the most useful context.",
  },
  learning: {
    label: "Capturing what helps next time",
    detail: "Saving helpful patterns for later.",
  },
  memory: {
    label: "Loading helpful memory",
    detail: "Bringing forward useful prior context.",
  },
  model_request: {
    label: "Requesting a stronger response",
    detail: "Calling for the best available response.",
  },
  preflight: {
    label: "Preparing your request",
    detail: "Checking readiness before moving forward.",
  },
  private_memos: {
    label: "Exploring your strongest options",
    detail: "Developing the best directions carefully.",
  },
  proof_workflow: {
    label: "Verifying every critical step",
    detail: "Checking each claim with care.",
  },
  reasoning: {
    label: "Working through your request",
    detail: "Developing the clearest path forward.",
  },
  reasoning_solver: {
    label: "Advancing your solution",
    detail: "Working through your best next steps.",
  },
  reasoning_solver_verifier: {
    label: "Confirming the final response",
    detail: "Giving the answer one last check.",
  },
  reconciliation: {
    label: "Aligning on the best answer",
    detail: "Bringing the strongest ideas together.",
  },
  research_program: {
    label: "Exploring the strongest evidence",
    detail: "Digging into the most relevant support.",
  },
  retry_round: {
    label: "Reworking the strongest answer",
    detail: "Tightening the response where needed.",
  },
  route: {
    label: "Selecting the strongest approach",
    detail: "Choosing the best path for you.",
  },
  socratic_review: {
    label: "Testing the key assumptions",
    detail: "Challenging weak spots before finalizing.",
  },
  solve: {
    label: "Building your answer",
    detail: "Turning the work into a response.",
  },
  synthesis: {
    label: "Bringing everything together",
    detail: "Combining the strongest parts clearly.",
  },
  synthesize: {
    label: "Preparing your final response",
    detail: "Shaping the answer for clarity.",
  },
  targeted_critiques: {
    label: "Pressure-testing the key ideas",
    detail: "Stress-testing the strongest directions.",
  },
  tool_broker: {
    label: "Choosing the right support",
    detail: "Deciding what extra help is needed.",
  },
  tool_workflow: {
    label: "Running focused support work",
    detail: "Collecting the right supporting results.",
  },
  triage: {
    label: "Selecting the strongest approach",
    detail: "Choosing the best path for you.",
  },
  validated_tool: {
    label: "Running focused support work",
    detail: "Collecting the right supporting results.",
  },
  verify: {
    label: "Checking the result",
    detail: "Confirming the work holds up.",
  },
  verifier: {
    label: "Confirming the final response",
    detail: "Giving the answer one last check.",
  },
  verifier_retry: {
    label: "Rechecking the final response",
    detail: "Verifying the revised answer again.",
  },
};

const TOOL_STATUS_COPY: Record<string, string> = {
  completed: "Integrating the latest results",
  failed: "Recovering and moving forward",
  started: "Running focused support work",
};

const normalizeToken = (value: string): string => value.trim().toLowerCase();

const getPhaseDisplayCopy = (phase: string): PhaseDisplayCopy | null => {
  const normalized = normalizeToken(phase);
  return normalized ? PHASE_DISPLAY_COPY[normalized] ?? null : null;
};

export const getPhaseLabel = (phase: string): string | null =>
  getPhaseDisplayCopy(phase)?.label ?? null;

export const getPhaseDetail = (phase: string): string | null =>
  getPhaseDisplayCopy(phase)?.detail ?? null;

export const getPhaseThinkingText = (phase: string): string | null => {
  const copy = getPhaseDisplayCopy(phase);
  return copy?.thinking ?? copy?.label ?? null;
};

export const getToolStatusThinkingText = (status: string): string | null => {
  const normalized = normalizeToken(status);
  return normalized ? TOOL_STATUS_COPY[normalized] ?? null : null;
};

export { DEFAULT_THINKING_TEXT };
