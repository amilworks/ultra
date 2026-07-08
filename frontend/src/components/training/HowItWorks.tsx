// The "How this works" disclosure (§14.6-G): the six-sentence loop description
// plus the glossary - the durable home of the in-place glosses.

export function HowItWorks() {
  return (
    <div className="training-gloss" style={{ display: "grid", gap: "0.5rem" }}>
      <p>Reviewed annotations sync automatically from BisQue into this model's training pool.</p>
      <p>When enough new reviewed data accumulates, a retrain can be requested — it never starts on its own.</p>
      <p>Every new candidate takes the same frozen exam: the gold set.</p>
      <p>
        A candidate that scores worse than the current model on any check is not promoted — your current model stays.
      </p>
      <p>A passing candidate first serves as a canary on 1 in 10 real runs.</p>
      <p>
        After the canary soaks without problems it can be promoted to active — and any promotion can be rolled back
        instantly.
      </p>
    </div>
  );
}
