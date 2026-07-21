<!--
  Thanks for the PR! Keep the summary tight and check the boxes below.
  The labeler bot auto-applies `area:` labels from your changed paths.
-->

## Summary

<!-- What does this change, and why? 1–3 sentences. -->

## Changes

-

## Testing

<!-- How did you verify this? Link runs / paste output / attach screenshots. -->

- [ ] Automated tests pass (Go: `make -C backend/controlplane test`; frontend: `pnpm --dir frontend test:unit`)
- [ ] Typecheck + lint clean (frontend: `pnpm --dir frontend typecheck && pnpm --dir frontend lint`)
- [ ] Verified locally (dev stack / browser) where the change is observable

## Checklist

- [ ] Added exactly one `type:` label and the relevant `area:` label(s)
- [ ] Updated docs / tests if behavior changed
- [ ] No secrets, credentials, or generated build artifacts committed
- [ ] Breaking changes are described above and tagged `breaking-change`
