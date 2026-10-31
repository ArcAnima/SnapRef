"""character_tags_v2new.jsonl -> character_tags_v2new_clean.jsonl for scenes_v2 (2026-10-31 smoke review).
Low-post characters pick up outfit tags that are not their look or are not acceptable for a character pool that
includes minors: changing clothes (dressing, *_pull, unzipped ...), revealing / swimwear, cosplay of someone else.
Those tags are dropped from the outfit list; a character keeps its hair/eye appearance tags. Characters with fewer
than 2 hair/eye tags left are dropped (Anima cannot hold them consistent, e.g. weather_report: hat + horns only)."""
import json, re, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
DROP = re.compile(r"(dressing|undress|changing|_pull\b|_pull$|pulled|_lift|lifted|unzip|unbutton|open_(shirt|jacket|coat|vest|"
                  r"clothes|kimono|robe|fly)|partially_unzipped|see-through|swimsuit|bikini|sarong|underwear|bra\b|_bra$|"
                  r"panties|lingerie|naked|nude|towel|bath|no_(pants|shirt|bra|panties)|bottomless|topless|torn_|"
                  r"off_shoulder|strapless|backless|bare_|cleavage|midriff|navel|thong|leotard|bodysuit|latex|pasties|"
                  r"nipple|censor|_aside|\(cosplay\)|^cosplay$|kigurumi|^costume$|bandaid$|bandaids)")
HAIR_EYES = re.compile(r"(hair|eyes|bangs|ponytail|twintails|braid|bun|ahoge|sidelocks|bob_cut|hime_cut)")
kept = dropped_chars = 0; dropped_tags = {}
with dst.open("w", encoding="utf-8", newline="\n") as out:
    for line in src.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        bad = [t for t, _ in x["outfit"] if DROP.search(t)]
        for t in bad: dropped_tags[t] = dropped_tags.get(t, 0) + 1
        x["outfit"] = [[t, f] for t, f in x["outfit"] if not DROP.search(t)]
        x["appearance"] = [[t, f] for t, f in x["appearance"] if not DROP.search(t)]
        if sum(bool(HAIR_EYES.search(t)) for t, _ in x["appearance"]) < 2:
            dropped_chars += 1; continue
        out.write(json.dumps(x, ensure_ascii=False) + "\n"); kept += 1
assert DROP.search("dressing") and DROP.search("shorts_pull") and DROP.search("cure_berry_(cosplay)") and DROP.search("black_bikini")
assert not DROP.search("black_thighhighs") and not DROP.search("pleated_skirt") and not DROP.search("superhero_costume")
print({"kept": kept, "dropped_chars": dropped_chars, "dropped_tags": sorted(dropped_tags.items(), key=lambda kv: -kv[1])[:30]})
