export function PanelLoadingState({
  title = "Loading panel...",
  subtitle = "Preparing this workspace only when you open it keeps the chat shell lighter.",
}: {
  title?: string;
  subtitle?: string;
}) {
  return (
    <div className="hero-state">
      <h2 className="hero-title">{title}</h2>
      <p className="hero-subtitle">{subtitle}</p>
    </div>
  );
}
