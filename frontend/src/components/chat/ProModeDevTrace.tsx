import { Check, Copy } from "lucide-react";

import { cn } from "@/lib/utils";

type RecordMap = Record<string, unknown>;

export type ProModeDevTraceProps = {
  messageId: string;
  conversation: RecordMap;
  isCopied: boolean;
  onCopy: (copyText: string) => void;
};

const toRecord = (value: unknown): RecordMap | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as RecordMap;
};

const toRecordArray = (value: unknown): RecordMap[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (item) => item && typeof item === "object" && !Array.isArray(item)
  ) as RecordMap[];
};

const proModeDevConversationCopyText = (conversation: RecordMap): string => {
  const markdown = String(conversation.markdown || "").trim();
  if (markdown) {
    return markdown;
  }
  const rounds = toRecordArray(conversation.rounds);
  const calculatorResults = toRecordArray(conversation.calculator_results);
  const verifier = toRecord(conversation.verifier);
  const synthesis = toRecord(conversation.synthesis);
  const sections: string[] = [];

  rounds.forEach((round, roundIndex) => {
    const roundNumber = Number(round.round_index ?? roundIndex + 1);
    const messages = toRecordArray(round.messages);
    const unresolvedCruxes = Array.isArray(round.unresolved_cruxes)
      ? round.unresolved_cruxes.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    const lines: string[] = [`Round ${roundNumber}`];
    if (unresolvedCruxes.length > 0) {
      lines.push(`Central cruxes: ${unresolvedCruxes.join("; ")}`);
    }
    messages.forEach((entry) => {
      const senderRole = String(entry.sender_role || "Unknown role").trim();
      const recipients = Array.isArray(entry.recipient_roles)
        ? entry.recipient_roles.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
      const vote = String(entry.vote || "").trim();
      const content = String(entry.content || "").trim();
      const objections = Array.isArray(entry.objections)
        ? entry.objections.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
      const requestedActions = Array.isArray(entry.requested_actions)
        ? entry.requested_actions.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
      lines.push(`- ${senderRole}${vote ? ` [${vote}]` : ""}`);
      if (recipients.length > 0) {
        lines.push(`  To: ${recipients.join(", ")}`);
      }
      if (content) {
        lines.push(`  ${content}`);
      }
      if (objections.length > 0) {
        lines.push(`  Objections: ${objections.join("; ")}`);
      }
      if (requestedActions.length > 0) {
        lines.push(`  Requested actions: ${requestedActions.join("; ")}`);
      }
    });
    sections.push(lines.join("\n"));
  });

  if (calculatorResults.length > 0) {
    sections.push(
      [
        "Calculator evidence",
        ...calculatorResults.map((item) => {
          const purpose = String(item.purpose || "Calculator check").trim();
          const status = item.success ? "Passed" : "Failed";
          const detail = String(
            item.formatted_result || item.error || item.expression || "No calculator result."
          ).trim();
          return `- ${purpose} [${status}]\n  ${detail}`;
        }),
      ].join("\n")
    );
  }

  if (synthesis) {
    sections.push(
      [
        "Synthesis",
        String(synthesis.response_text || "").trim() || "No synthesis draft recorded.",
      ].join("\n")
    );
  }

  if (verifier) {
    const issues = Array.isArray(verifier.issues)
      ? verifier.issues.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    sections.push(
      [
        "Verifier",
        verifier.passed ? "Passed" : "Flagged issues",
        ...issues.map((item) => `- ${item}`),
      ].join("\n")
    );
  }

  return sections.join("\n\n").trim() || JSON.stringify(conversation, null, 2);
};

export function ProModeDevTrace({
  messageId,
  conversation,
  isCopied,
  onCopy,
}: ProModeDevTraceProps) {
  const rounds = toRecordArray(conversation.rounds);
  const calculatorResults = toRecordArray(conversation.calculator_results);
  const verifier = toRecord(conversation.verifier);
  const synthesis = toRecord(conversation.synthesis);
  if (rounds.length === 0 && calculatorResults.length === 0 && !verifier && !synthesis) {
    return null;
  }
  return (
    <details className="pro-mode-dev-trace" data-testid="pro-mode-dev-trace">
      <summary>
        <span>Internal Pro Mode conversation (development only)</span>
        <button
          type="button"
          className={cn("pro-mode-dev-copy-button", isCopied && "pro-mode-dev-copy-button--copied")}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onCopy(proModeDevConversationCopyText(conversation));
          }}
          aria-label={isCopied ? "Copied internal Pro Mode conversation" : "Copy internal Pro Mode conversation"}
        >
          {isCopied ? (
            <>
              <Check className="size-3.5" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy className="size-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </summary>
      <div className="pro-mode-dev-trace-content">
        {rounds.map((round, roundIndex) => {
          const roundNumber = Number(round.round_index ?? roundIndex + 1);
          const messages = toRecordArray(round.messages);
          const unresolvedCruxes = Array.isArray(round.unresolved_cruxes)
            ? round.unresolved_cruxes.map((item) => String(item || "").trim()).filter(Boolean)
            : [];
          return (
            <section key={`${messageId}-round-${roundNumber}`} className="pro-mode-dev-round">
              <header className="pro-mode-dev-round-header">
                <h4>{`Round ${roundNumber}`}</h4>
                {unresolvedCruxes.length > 0 ? (
                  <span className="pro-mode-dev-round-status">
                    {`${unresolvedCruxes.length} central ${
                      unresolvedCruxes.length === 1 ? "crux" : "cruxes"
                    }`}
                  </span>
                ) : (
                  <span className="pro-mode-dev-round-status">No central blockers</span>
                )}
              </header>
              {messages.map((entry, entryIndex) => {
                const senderRole = String(entry.sender_role || "Unknown role").trim();
                const recipients = Array.isArray(entry.recipient_roles)
                  ? entry.recipient_roles.map((item) => String(item || "").trim()).filter(Boolean)
                  : [];
                const objections = Array.isArray(entry.objections)
                  ? entry.objections.map((item) => String(item || "").trim()).filter(Boolean)
                  : [];
                const requestedActions = Array.isArray(entry.requested_actions)
                  ? entry.requested_actions.map((item) => String(item || "").trim()).filter(Boolean)
                  : [];
                const content = String(entry.content || "").trim();
                const vote = String(entry.vote || "").trim();
                return (
                  <article
                    key={`${messageId}-round-${roundNumber}-message-${entryIndex}`}
                    className="pro-mode-dev-message"
                  >
                    <div className="pro-mode-dev-message-header">
                      <strong>{senderRole}</strong>
                      <span>{vote ? `Vote: ${vote}` : "Vote unavailable"}</span>
                    </div>
                    {recipients.length > 0 ? (
                      <p className="pro-mode-dev-message-meta">
                        {`To: ${recipients.join(", ")}`}
                      </p>
                    ) : null}
                    {content ? <p className="pro-mode-dev-message-body">{content}</p> : null}
                    {objections.length > 0 ? (
                      <div className="pro-mode-dev-message-list">
                        <span>Objections</span>
                        <ul>
                          {objections.map((item) => (
                            <li key={`${messageId}-${roundNumber}-${senderRole}-objection-${item}`}>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {requestedActions.length > 0 ? (
                      <div className="pro-mode-dev-message-list">
                        <span>Requested actions</span>
                        <ul>
                          {requestedActions.map((item) => (
                            <li key={`${messageId}-${roundNumber}-${senderRole}-action-${item}`}>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </section>
          );
        })}
        {calculatorResults.length > 0 ? (
          <section className="pro-mode-dev-round">
            <header className="pro-mode-dev-round-header">
              <h4>Calculator evidence</h4>
            </header>
            {calculatorResults.map((item, index) => (
              <article
                key={`${messageId}-calculator-${index}`}
                className="pro-mode-dev-message pro-mode-dev-evidence"
              >
                <div className="pro-mode-dev-message-header">
                  <strong>{String(item.purpose || "Calculator check").trim()}</strong>
                  <span>{item.success ? "Passed" : "Failed"}</span>
                </div>
                <p className="pro-mode-dev-message-body">
                  {String(
                    item.formatted_result || item.error || item.expression || "No calculator result."
                  ).trim()}
                </p>
              </article>
            ))}
          </section>
        ) : null}
        {synthesis ? (
          <section className="pro-mode-dev-round">
            <header className="pro-mode-dev-round-header">
              <h4>Synthesis</h4>
            </header>
            <article className="pro-mode-dev-message">
              <p className="pro-mode-dev-message-body">
                {String(synthesis.response_text || "").trim() || "No synthesis draft recorded."}
              </p>
            </article>
          </section>
        ) : null}
        {verifier ? (
          <section className="pro-mode-dev-round">
            <header className="pro-mode-dev-round-header">
              <h4>Verifier</h4>
              <span className="pro-mode-dev-round-status">
                {verifier.passed ? "Passed" : "Flagged issues"}
              </span>
            </header>
            {Array.isArray(verifier.issues) && verifier.issues.length > 0 ? (
              <article className="pro-mode-dev-message">
                <div className="pro-mode-dev-message-list">
                  <span>Issues</span>
                  <ul>
                    {verifier.issues.map((item, index) => (
                      <li key={`${messageId}-verifier-issue-${index}`}>
                        {String(item || "").trim()}
                      </li>
                    ))}
                  </ul>
                </div>
              </article>
            ) : null}
          </section>
        ) : null}
      </div>
    </details>
  );
}
