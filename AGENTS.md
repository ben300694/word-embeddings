# AGENTS.md

Guidance for agents working with this teaching-material repository.

## Repository context

- This repository contains material for the HHU master seminar "Word Embedding Spaces", taught in Summer Terms 2022
  and 2023.
- The seminar is complete. Preserve the material as an archive; do not treat it as an actively maintained course
  unless the user explicitly asks to resume development.
- Preserve unrelated files and contributor history. Several course materials were contributed by other seminar
  participants.

## Git remotes and archival status

- Before changing or synchronizing remotes, read
  [`docs/repository-remotes.md`](docs/repository-remotes.md).
- Keep GitHub remote `origin` as the public source repository.
- Treat remote `gitlab` as the private HHU DSML teaching archive.
- The GitLab copy is a completed one-time archive. It does not require another routine synchronization in September
  unless the GitHub repository changes after the verified archive snapshot.
- Do not use `git push --mirror`; synchronize explicitly selected branches and tags so implementation refs cannot be
  copied or archival refs deleted accidentally.

