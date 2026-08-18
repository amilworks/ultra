# Product font provenance

Ultra bundles **Ultra Sans** as its primary product face and retains the
unmodified official Inter Variable v4.1 webfonts as the first coverage fallback.
Both families are local WOFF2 assets; the application makes no network request
to a font host.

## Ultra Sans — primary product face

Ultra Sans was designed and authored by **Amil Khan**, a PhD student in the
Department of Electrical and Computer Engineering at the University of
California, Santa Barbara, for **Ultra**, an agentic system for science. The
family starts from DM Sans and incorporates adapted Inter glyph geometry. Full
construction, authorship, and upstream lineage are maintained in the dedicated
[Ultra Sans repository](https://github.com/amilworks/ultra-sans).

This application pins the last font-producing development snapshot at
[`717ad23b67802f2e2d521e566ca3d390b48a83c1`](https://github.com/amilworks/ultra-sans/tree/717ad23b67802f2e2d521e566ca3d390b48a83c1).
That source revision reproducibly emits these exact artifacts:

| Asset | Style | Variable axes | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `UltraSans-Variable.woff2` | normal | `wght` 100–1000; `opsz` 9–40 | 126880 | `f060de034541b34034450670bc9becf7c0640f57f2c23dff311ca04a7ff5c97d` |
| `UltraSans-Italic-Variable.woff2` | italic | `wght` 100–1000; `opsz` 9–40 | 154524 | `26470a9271f845356cfd113a15e5df9e623d440bacab5498e45ad16051e5771d` |

Status: development snapshot, not yet a stable public font release. Updating
either binary requires updating its pinned source revision, byte count, digest,
and the typography contract in the same change.

Ultra Sans is distributed under the SIL Open Font License 1.1. The verbatim DM
Sans notice is preserved as `OFL-DM-Sans.txt`. The adapted Inter data is covered
by the Inter notice preserved as `OFL-1.1.txt`; the generated binaries retain
the upstream copyright and license records in their embedded metadata.

## Inter 4.1 — coverage fallback

Inter remains after Ultra Sans in the CSS family stack so characters outside
the current Ultra Sans map still receive a metrically compatible local face.
It is not the product's primary UI or reading face.

| Asset | Style | Variable axes | Official source | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `InterVariable-v4.1.woff2` | normal | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable.woff2?v=4.1` | 352240 | `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3` |
| `InterVariable-Italic-v4.1.woff2` | italic | `wght` 100–900; `opsz` 14–32 | `https://rsms.me/inter/font-files/InterVariable-Italic.woff2?v=4.1` | 387976 | `e564f652916db6c139570fefb9524a77c4d48f30c92928de9db19b6b5c7a262a` |

Version: Inter 4.1. License: SIL Open Font License 1.1; the upstream
`LICENSE.txt` for tag `v4.1` is preserved verbatim as `OFL-1.1.txt`.

Only WOFF2 is shipped. We deliberately do not preload a font today: the benefit
has not yet been measured against the extra early-bandwidth cost on Ultra's
scientific workspaces. If future production measurements justify preloading,
only the Ultra Sans normal face may be considered. Both italic faces must remain
demand-loaded. The post-build typography contract requires any future preload
URL to exactly match the emitted Ultra Sans normal-face URL.
