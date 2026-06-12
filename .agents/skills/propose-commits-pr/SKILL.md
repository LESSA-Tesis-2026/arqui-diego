---
name: propose-commits-pr
description: Propose a progressive conventional-commit plan and PR title/description for current unstaged changes. Use when the user wants a commit strategy before approving commits, especially when changes must be split safely and some files must not be touched.
---

# Propose Commits And PR

Use this skill to inspect the current worktree and propose a safe, progressive commit plan plus a PR title and PR description. This skill does **not** create commits, stage files, modify files, or create pull requests. It prepares the plan and waits for explicit approval.

## Purpose

Create a practical reviewable commit strategy for the current unstaged changes. The output must help the user understand:

- Which commits should be created.
- Which files belong in each commit.
- Which files must be excluded.
- What PR title and PR body should be used.

## Required Style

- Use Conventional Commits.
- Include git emojis in commit messages and PR title.
- Keep commit messages focused and scoped.
- Prefer progressive commits that map to logical review units.
- Keep the PR description useful for reviewers.
- Do not include the PR title inside the PR description.
- Do not include useless PR sections such as `Notes`, `Verification`, or generic boilerplate unless the user explicitly asks for them.
- Keep the PR description focused on the actual functional areas changed.

## Safety Rules

- Never stage files.
- Never create commits.
- Never amend commits.
- Never create or update a PR.
- Never modify files as part of this skill.
- Always wait for explicit user approval before any commit operation.
- Do not include files the user explicitly said not to touch.
- Treat local `.env` files, secrets, credentials, virtual environments, build outputs, caches, and generated artifacts as excluded by default.
- If `.gitignore` has user changes and the user says not to touch it, explicitly exclude it.
- If unrelated tracked changes exist, identify them and exclude them unless the user asks otherwise.

## Inspection Workflow

Run these git commands only for inspection:

```bash
git status --short
git diff --stat
git log --oneline -5
```

Use recent commit messages to match repository style. If needed, inspect file paths with non-destructive file listing/search tools to understand the scope, but do not read secrets or local env files.

## Proposal Workflow

1. Summarize the current worktree categories.
2. Identify files that must be excluded.
3. Split included files into logical commits.
4. For each commit, provide:
   - Commit message.
   - Included paths.
   - Excluded paths if relevant.
   - Short reason for the split.
5. Provide a PR title using Conventional Commit style with git emoji.
6. Provide a PR description that includes only useful reviewer context.
7. End by explicitly stating that no commits will be created until the user approves.

## Recommended Output Shape

```markdown
**Commit Proposal**

Exclude from all commits:

- `<path>`
- `<path>`

**Commit 1**

```text
feat(scope): ✨ concise message
```

Include:

```text
path/**
path/file.ext
```

Why: <short reason>

**Commit 2**

```text
docs(scope): 📝 concise message
```

Include:

```text
path/file.ext
```

Why: <short reason>

**PR Title**

```text
feat(scope): ✨ concise title
```

**PR Description**

```markdown
## Summary

- <useful reviewer-facing summary>
- <useful reviewer-facing summary>

## Backend

- <if applicable>

## Frontend

- <if applicable>
```

I will wait for approval before creating commits.
```

## Example For Current Project

When the worktree contains the LESSA API/web app work, a good split is:

```text
feat(api): ✨ add LESSA model serving gateway
feat(web): ✨ add LESSA translation experience
docs(dev): 📝 document local app workflows
```

Exclude:

```text
.gitignore
modeloTutorialFtGemini/config.py
apps/api/.env
apps/web/.env
apps/api/.venv/**
apps/web/node_modules/**
apps/web/.next/**
**/__pycache__/**
**/.pytest_cache/**
```
