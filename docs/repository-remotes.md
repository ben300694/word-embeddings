# Repository remotes and HHU teaching archive

Procedure reviewed: 2026-09-01. This page intentionally contains durable workflow rather than a moving archive
snapshot.

This repository deliberately uses two independent Git hosts:

| Remote | Role | URL |
| --- | --- | --- |
| `origin` | Public GitHub source repository | `git@github.com:ben300694/word-embeddings.git` |
| `gitlab` | Private HHU DSML teaching archive | `git@gitlab.cs.uni-duesseldorf.de:dsml/teaching/word-embedding-spaces-seminar-2022-2023.git` |

Project pages:

- GitHub: <https://github.com/ben300694/word-embeddings>
- HHU GitLab: <https://gitlab.cs.uni-duesseldorf.de/dsml/teaching/word-embedding-spaces-seminar-2022-2023>
  (project ID 3732)

The GitLab project is named **Word Embedding Spaces Seminar - Summer Terms 2022-2023**. It is private and inherits
access from the `dsml/teaching` subgroup. GitHub remains public.

## Synchronization model

The remotes are not automatically mirrored. The seminar is complete, so the GitLab synchronization performed on
2026-08-31 is intended as the final archive snapshot. No routine September synchronization is needed unless a later
commit changes the GitHub repository.

The archived Git content consists of the single `main` branch and its complete history. There were no Git tags to
copy. GitHub issues, pull-request discussions, releases, Actions history, and repository settings are not transferred
by a Git push.

Changing ref targets, storage figures, and membership snapshots are tracked in the current maintainer's Housekeeping
dashboard rather than duplicated here. Re-read both live remotes and the GitLab project before any later update.

If a later update is intentionally made, verify and synchronize it explicitly:

```sh
git fetch origin --prune --tags
git ls-remote --heads origin
git ls-remote --tags --refs origin
git ls-remote --heads gitlab
git ls-remote --tags --refs gitlab
origin_main_object_id="$(git rev-parse refs/remotes/origin/main)"
test "$origin_main_object_id" = "$(git ls-remote origin refs/heads/main | awk '{print $1}')"
git push gitlab "$origin_main_object_id:refs/heads/main"
# Repeat for each reviewed live GitHub tag after replacing TAG_NAME:
origin_tag_object_id="$(git rev-parse refs/tags/TAG_NAME)"
test "$origin_tag_object_id" = "$(git ls-remote --tags --refs origin refs/tags/TAG_NAME | awk '{print $1}')"
git push gitlab "$origin_tag_object_id:refs/tags/TAG_NAME"
git ls-remote origin refs/heads/main
git ls-remote gitlab refs/heads/main
```

Replace `TAG_NAME` rather than running it literally. The variables freeze objects that the preceding fetch made
available locally; the `test` commands refuse to push if the corresponding live GitHub ref changed afterward. Using
`git rev-parse refs/tags/TAG_NAME` preserves an annotated tag object instead of accidentally substituting its peeled
commit. Apply the same freeze, live-ref check, and explicit mapping to every additional intended branch. Avoid
`git push --all`, a broad `git push --tags`, and `git push --mirror`: they can copy implementation refs or delete
archival refs without an individual review.
