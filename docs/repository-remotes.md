# Repository remotes and HHU teaching archive

Last verified: 2026-08-31.

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

If a later update is intentionally made, verify and synchronize it explicitly:

```sh
git fetch origin --prune --tags
git push gitlab refs/remotes/origin/main:refs/heads/main
git push gitlab --tags
git ls-remote origin refs/heads/main
git ls-remote gitlab refs/heads/main
```

Avoid `git push --mirror`: it can copy implementation refs or delete archival refs that are absent locally.

