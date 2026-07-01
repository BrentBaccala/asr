# Vendored flag SVGs — source & license

These 13 SVGs are the `4x3` flags from **flag-icons** by Panayiotis
Lipiridis (lipis):

- Project: https://github.com/lipis/flag-icons
- npm package: `flag-icons` (installed **v7.5.0**)
- Files taken from: `node_modules/flag-icons/flags/4x3/<cc>.svg`
- License: **MIT** — see `LICENSE` in this directory (full text vendored
  as MIT requires the notice to accompany the files). The flag artwork
  itself is public domain per the project.

Vendored (rather than depending on npm at build time) so
`web/sign/build.py` runs offline with no `node_modules`.

Country codes and the language each represents on the sign:

| File | Country | Language on sign |
|------|---------|------------------|
| `us.svg` | United States | English |
| `es.svg` | Spain | Spanish |
| `it.svg` | Italy | Italian |
| `fr.svg` | France | French |
| `de.svg` | Germany | German |
| `pt.svg` | Portugal | Portuguese |
| `nl.svg` | Netherlands | Dutch |
| `ru.svg` | Russia | Russian |
| `sa.svg` | Saudi Arabia | Arabic |
| `in.svg` | India | Hindi |
| `cn.svg` | China | Chinese |
| `jp.svg` | Japan | Japanese |
| `kr.svg` | South Korea | Korean |

For languages spoken across many countries the flag is a judgement call
(English→US, Spanish→Spain, Portuguese→Portugal, Arabic→Saudi Arabia);
edit the `ROWS` table in `../build.py` to change any mapping.

To refresh from upstream:

```
npm install flag-icons
cp node_modules/flag-icons/flags/4x3/{us,es,it,fr,de,pt,nl,ru,sa,in,cn,jp,kr}.svg .
cp node_modules/flag-icons/LICENSE .
```
