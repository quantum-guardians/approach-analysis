# Release Notes Workflow

## Why

Release notes are read by whoever installs or runs this version. They should
describe what changed for that person, not how the change was produced.

## Audience

Write for the consumer of the build, not the reviewer of the code. Development
narrative belongs in the pull request; planning belongs in the issue.

## Body Template

Write the notes in Korean. Keep code identifiers, API names, CLI commands, and
error strings in their original form.

```markdown
한 줄 요약.

## 변경
- 사용자가 체감하는 변화

## 수정
- 고친 결함

## 주의
- 이전 버전과 달라져 미리 알아야 할 것

전체 변경: #123
```

Drop any heading with no content. When fewer than about three items changed,
keep the summary line and link the pull request instead of adding headings.

## Do Not Include

- **Build duration or CI progress** — untrue within minutes of publishing.
- **Plans for the next version, or remaining work** — belongs in that version's
  notes.
- **Process narrative, or what was left unverified** — belongs in the pull
  request.
- **Internal symbol names** — translate into the observable change.

## Generated Sections

Release automation often appends content to the notes after builds finish:
download tables, checksums, install instructions, hosted links. These are
usually delimited by markers such as `<!-- links:start -->`.

Treat generated blocks as owned by the automation. Do not hand-write them, and
do not remove them.

Confirm whether such automation exists before editing published notes. A job
that appends on publish will not run again for an already-built release, so
content removed by hand is lost until restored by hand.

## Safe Editing

`gh release edit --notes` and `--notes-file` both replace the entire body, not
just the part you wrote. Editing published notes without preserving generated
blocks silently deletes them.

Write the intended body to a Markdown file and pass it with `--notes-file`; do
not interpolate multiline content into `--notes`. This prevents shell quoting,
command substitution, and newline damage:

```bash
gh release edit "$TAG" --notes-file /tmp/release-notes.md
```

When the release already carries a generated block, read the current body first
and keep that block verbatim in the new file.

After editing, read the remote release back with `gh release view` and confirm
the body matches the source file, including any generated block. Fix the remote
release before reporting completion.

## Version Consistency

Keep the tag and the version recorded inside the project in sync. Release
automation usually reads only the tag, while the built artifact reports the
in-project value, so a bump in one place does not update the other.

Bump the in-project version in its own commit before tagging, and tag that
commit.

## Before Publishing

- Confirm the tag points at the intended commit.
- Confirm the in-project version matches the tag.
- Confirm required builds succeeded when publishing gates a build.
- Confirm notes contain no process narrative or forward-looking plans.
