#!/usr/bin/env bash
#
# Fetch model weights and scene data into the paths the shipped configs expect.
# See assets/MANIFEST.md for what each file is and how large it is.
#
#   scripts/download_assets.sh                # weights + the 500-scene eval set
#   scripts/download_assets.sh weights        # weights only
#   scripts/download_assets.sh eval-scenes    # the 500-scene eval set only
#   scripts/download_assets.sh all            # adds maps and demonstrations
#                                             # (~2.9 GB down, 47 GB on disk)
#
# Re-running is safe: files already present with the right checksum are skipped.

set -euo pipefail

# ---------------------------------------------------------------------------
# Assets are attached to this repository's GitHub Release. Remote filenames are
# flat and must match exactly; local destinations are nested and created here.
#
# Host them elsewhere by pointing SOGUDIFF_ASSET_URL at any flat-namespace
# location, keeping the same filenames:
#   GitHub Release   https://github.com/<user>/<repo>/releases/download/<tag>
#   Zenodo record    https://zenodo.org/api/records/<id>/files
#   Plain web server https://example.org/sogudiff
#
# If the release is not public (a private repo, or a draft), set GITHUB_TOKEN
# with read access as well: release assets ignore a token on the plain download
# URL, so this script resolves them through the API instead.
# ---------------------------------------------------------------------------
BASE_URL="${SOGUDIFF_ASSET_URL:-https://github.com/schaiblc/SoGuDiff/releases/download/v1.0}"

# Zenodo's API serves a file's bytes at <base>/<name>/content; other hosts at
# <base>/<name>.
case "$BASE_URL" in
    *zenodo.org/api/*) URL_SUFFIX="/content" ;;
    *)                 URL_SUFFIX="" ;;
esac

# A private GitHub repo needs the API asset endpoint plus a token: the plain
# releases/download URL ignores credentials and returns HTML. Resolve asset ids
# once, up front, so the rest of the script just fetches URLs.
declare -A ASSET_URL=()
if [ -n "${GITHUB_TOKEN:-}" ] && [[ "$BASE_URL" == *github.com/*/releases/download/* ]]; then
    _path="${BASE_URL#*github.com/}"; _owner_repo="${_path%%/releases/*}"
    _tag="${BASE_URL##*/releases/download/}"
    echo "GITHUB_TOKEN set: trying the API for $_owner_repo@$_tag (needed only if the release is private)"
    _hdr=(-H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json")
    # Published releases resolve by tag. A DRAFT has no git tag yet, so fall
    # back to listing releases, which includes drafts for anyone with push
    # access, and match on tag_name.
    _json=$(curl -sf "${_hdr[@]}" \
        "https://api.github.com/repos/$_owner_repo/releases/tags/$_tag" 2>/dev/null || true)
    if [ -z "$_json" ]; then
        _json=$(curl -sf "${_hdr[@]}" \
            "https://api.github.com/repos/$_owner_repo/releases?per_page=100" 2>/dev/null \
          | python3 -c 'import json,sys
tag=sys.argv[1]
for r in json.load(sys.stdin):
    if r.get("tag_name")==tag: print(json.dumps(r)); break' "$_tag" 2>/dev/null || true)
        [ -n "$_json" ] && echo "  (found it as a draft release)"
    fi
    if [ -z "$_json" ]; then
        # Not fatal: a public release downloads fine without any of this, and
        # plenty of people have GITHUB_TOKEN exported for unrelated reasons.
        echo "  (token cannot see that release; continuing with public URLs)"
    else
        while IFS=$'\t' read -r _n _u; do ASSET_URL["$_n"]="$_u"; done < <(
            python3 -c 'import json,sys
d=json.load(sys.stdin)
for a in d.get("assets", []): print(a["name"], a["url"], sep="\t")' <<<"$_json")
        echo "  resolved ${#ASSET_URL[@]} assets via the API"
    fi
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIER="${1:-default}"

# "<url path>|<destination>|<sha256>" — sha256 may be "-" to skip verification.
WEIGHTS=(
  "sogudiff_single_axis.pt|checkpoints/sogudiff_single_axis.pt|0b03fdacbc5762f611d3522a5ba999d7dca0c3f1232902b1114f8fc11e125687"
  "sogudiff_joint.pt|checkpoints/sogudiff_joint.pt|de55dc4de4b47b680cc1c400345c8ee171d107248fd081c219553c593978bb06"
  "dsrnn.pt|baselines/dsrnn/data/trained/checkpoints/best.pt|9a2beb25525cc85ea79d21f5795be54b9df6f4c928714548a46684797a7773c2"
  "navistar.pt|baselines/navistar/data/navigation/star_sac/checkpoints/best_sac_actor.pt|6d4c1c1980851b9b626756cc2b5e1cac3f2f489d5e1140770832a6184f21adf6"
  "height.pt|baselines/height/data/trained/checkpoints/best.pt|05891d2763e1a15ecc4eb3f47fa8bfe3bbe23fab74a5504514c5d9f2d3c1cb4e"
  "cadrl_il_model.pth|crowdnav_env/crowd_nav/models/cadrl/il_model.pth|b26aa25d579d4d1e081107d2ea4341c9beeb75fb702cdcb0fb1aefc7c9c642c2"
  "cadrl_rl_model.pth|crowdnav_env/crowd_nav/models/cadrl/rl_model.pth|3ff2879e4f8d0da41dceca8d183a9256fe3e8e627f736fdd843f1aff3956bb71"
  "lstm_rl_il_model.pth|crowdnav_env/crowd_nav/models/lstm_rl/il_model.pth|abcee3e492f1a5646dc2771d051bfd1be6a5cff173a8bedccefb4261fbbc1361"
  "lstm_rl_rl_model.pth|crowdnav_env/crowd_nav/models/lstm_rl/rl_model.pth|18d9d497a8f65d08e07ce0e76137f8ee269ab74b2b874b0d2268ecea14aa2d58"
  "sarl_il_model.pth|crowdnav_env/crowd_nav/models/sarl/il_model.pth|66a3b45f51a98a5274d1725ffa5c5d33528f84d2446d6375abc5d5d761d923eb"
  "sarl_rl_model.pth|crowdnav_env/crowd_nav/models/sarl/rl_model.pth|d98cd273077e88495bdfc258d2f3680e36796ca807a5a3b387e92aae7e07fa34"
  "rgl_il_model.pth|crowdnav_env/crowd_nav/models/rgl/il_model.pth|b3cbb934744f7863b5b4179fbe44cd0b3ac770649dd6f466f5ac8bd0daedef05"
  "rgl_rl_model.pth|crowdnav_env/crowd_nav/models/rgl/rl_model.pth|7a106c7197bb6ec13b62f7de6f194d9c5481e4c8f2cc57526efc0bb517a8c889"
)
ARCHIVES_EVAL=(
  "eval_500.tar.zst|data/scenes/eval_500|8e9a7a6c3f3db2ecb4459cecf44858102e36d53ec9a342f38f2b65ec94d86a8b"
)
# Demonstrations are split one archive per map source: each is a manageable
# download on its own, and training can use any subset of them via
# --dataset_dirs.
ARCHIVES_FULL=(
  "maps.tar.zst|data/maps|53351d7e00d375b79cc6db0f6bad054a843f23bc0aac60bfbb869a0f53af6668"
  "expert_custom.tar.zst|data/expert/custom|6be2acc7c3feee91a843b4365d8babf3a8cc22124759c9820cb7ea4d78e628fe"
  "expert_interiorgs.tar.zst|data/expert/interiorgs|d907cc4534841c9f28e0a1b35f094ac167cb10513b79f92549463304f42623c1"
  "expert_matterport.tar.zst|data/expert/matterport|6481555ec100e8a7b1d0db3183c155d626c29c8eaffde194e06fa184fb69f17e"
  "expert_tartanground.tar.zst|data/expert/tartanground|1b3bd443d07581f468ec8ab470074c4a112d8ca1c4efd62ec5ccaffad0fe9c83"
)

have() { command -v "$1" >/dev/null 2>&1; }

fetch() {   # fetch <remote name> <destination>
    # Declared in two statements on purpose: a single `local` expands all of
    # its words before assigning any of them, so referencing $name in the same
    # statement that defines it dies under `set -u`.
    local name="$1" dest="$2"
    local url="$BASE_URL/$name$URL_SUFFIX"
    local -a auth=()
    if [ -n "${ASSET_URL[$name]:-}" ]; then          # private GitHub release
        url="${ASSET_URL[$name]}"
        auth=(-H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream")
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  -> $dest"
    local rc=0
    if have curl; then
        curl -fL --retry 3 --retry-delay 5 -C - "${auth[@]}" -o "$dest.part" "$url" || rc=$?
    elif have wget; then
        [ ${#auth[@]} -gt 0 ] && { echo "ERROR: a private release needs curl." >&2; exit 1; }
        wget -c -O "$dest.part" "$url" || rc=$?
    else
        echo "ERROR: neither curl nor wget is available." >&2; exit 1
    fi
    if [ "$rc" -ne 0 ]; then
        rm -f "$dest.part"
        echo "ERROR: could not download '$name' (exit $rc)." >&2
        echo "       URL: $url" >&2
        if [ ${#auth[@]} -eq 0 ]; then
            echo "       If the release is private or a draft, export a GITHUB_TOKEN" >&2
            echo "       with read access and re-run; the script then fetches via the" >&2
            echo "       GitHub API. Otherwise check that the release and this asset" >&2
            echo "       name exist, or override the host with SOGUDIFF_ASSET_URL." >&2
        else
            echo "       A token was used, so it may lack read access to this repo," >&2
            echo "       or the asset name may not match what the release carries." >&2
        fi
        exit 1
    fi
    mv -f "$dest.part" "$dest"
}

verify() {  # verify <file> <sha256>  -- returns non-zero on mismatch
    local file="$1" want="$2"
    [ "$want" = "-" ] && return 0
    have sha256sum || return 0
    [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ]
}

get_file() {
    local entry="$1"
    IFS='|' read -r path dest sha <<< "$entry"
    if [ -f "$dest" ] && verify "$dest" "$sha"; then
        echo "  ok   $dest (already present)"; return
    fi
    fetch "$path" "$dest"
    # Delete the bad file rather than leaving it where the configs point: the
    # whole purpose of the checksum is to stop a truncated download being used
    # as if it were a model.
    verify "$dest" "$sha" || {
        rm -f "$dest"
        echo "ERROR: checksum mismatch for $dest (removed; re-run to retry)" >&2
        exit 1
    }
}

get_archive() {
    local entry="$1"
    IFS='|' read -r path dest sha <<< "$entry"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  ok   $dest/ (already present)"; return
    fi
    local tmp="$dest.tar.zst"
    fetch "$path" "$tmp"
    verify "$tmp" "$sha" || {
        rm -f "$tmp"
        echo "ERROR: checksum mismatch for $tmp (removed; re-run to retry)" >&2
        exit 1
    }
    echo "  unpacking $tmp"
    mkdir -p "$dest"
    if have zstd; then
        tar --use-compress-program=unzstd -xf "$tmp" -C "$dest" --strip-components=1
    else
        echo "ERROR: zstd is required to unpack $tmp (apt install zstd)." >&2; exit 1
    fi
    rm -f "$tmp"
}

case "$TIER" in
  weights)
    echo "== model weights =="
    for e in "${WEIGHTS[@]}"; do get_file "$e"; done ;;
  eval-scenes)
    echo "== 500-scene evaluation set =="
    for e in "${ARCHIVES_EVAL[@]}"; do get_archive "$e"; done ;;
  all)
    echo "== model weights =="
    for e in "${WEIGHTS[@]}"; do get_file "$e"; done
    echo "== scenes and datasets (large) =="
    for e in "${ARCHIVES_EVAL[@]}" "${ARCHIVES_FULL[@]}"; do get_archive "$e"; done ;;
  default)
    echo "== model weights =="
    for e in "${WEIGHTS[@]}"; do get_file "$e"; done
    echo "== 500-scene evaluation set =="
    for e in "${ARCHIVES_EVAL[@]}"; do get_archive "$e"; done ;;
  *)
    echo "usage: $0 [weights|eval-scenes|all]" >&2; exit 2 ;;
esac

echo
echo "Done. Verify with:"
echo "  cd crowdnav_env/crowd_nav"
if [ "$TIER" = "weights" ]; then
    # The saved scene set was not fetched; use the procedural benchmark, which
    # is also what the paper's comparison table uses.
    echo "  python evaluate.py --policy sogudiff \\"
    echo "      --infer_mode per_axis --w_normalize --gpu --no_video"
else
    echo "  python evaluate.py --policy sogudiff \\"
    echo "      --infer_mode per_axis --w_normalize --gpu --no_video \\"
    echo "      --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 20"
fi
