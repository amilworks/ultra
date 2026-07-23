# Inter Variable font provenance

Ultra vendors the unmodified official Inter Variable v4.1 webfont assets.

| Asset | Style | Variable axes | Official source | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `InterVariable-v4.1.woff2` | normal | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable.woff2?v=4.1` | 352240 | `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3` |
| `InterVariable-Italic-v4.1.woff2` | italic | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable-Italic.woff2?v=4.1` | 387976 | `e564f652916db6c139570fefb9524a77c4d48f30c92928de9db19b6b5c7a262a` |

Version: Inter 4.1. License: SIL Open Font License 1.1; the upstream
`LICENSE.txt` for tag `v4.1` is preserved verbatim as `OFL-1.1.txt`.

Only WOFF2 is shipped. We deliberately do not preload a font today: the benefit
has not yet been measured against the extra early-bandwidth cost on Ultra's
scientific workspaces. If future production measurements justify preloading,
only the normal face may be considered. The italic face must remain
demand-loaded. The post-build typography contract requires any future preload
URL to exactly match the emitted normal-face URL.
