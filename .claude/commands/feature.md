---
description: Plan, implement and review a feature with subagents
argument-hint: <feature description>
---
Feature: $ARGUMENTS

Follow this loop:
1. Use the `planner` subagent to produce a plan. Show me a short summary and WAIT for my 'ok'.
2. Use the `implementer` subagent to implement exactly the approved plan on a new branch.
3. Use the `reviewer` subagent on the result.
4. If the verdict is CHANGES_REQUESTED, send the BLOCKING items back to the `implementer`. Maximum 2 rounds. If it still fails, stop and summarize the problem for me.
5. When tests and lint pass and the verdict is APPROVE: commit with a Conventional Commit message and open a PR with `gh pr create`.
