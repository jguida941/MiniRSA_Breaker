## Summary

- [ ] Linked issue or short description of the change and why it is needed.

## Quality Gates (tick once complete)

- [ ] `pre-commit run --all-files` (Ruff, Ruff-format, Black, mypy, Bandit, optional pip-audit) passes locally.
- [ ] `make quality` (or equivalent Ruff → mypy → pytest → bandit → pip-audit) passes.
- [ ] Targeted tests for modified areas run locally (`pytest -k ...`, `make gui`, etc.).
- [ ] Mutation scope (if core logic changed): `make mutation MUTATE_PATHS=...` run and survivors reviewed.
- [ ] Documentation updated (README/Playbook/Contributing) if behavior or workflow changed.

## Testing

- [ ] `pytest -q` (non-GUI suite)
- [ ] `pytest -m gui` (if GUI elements changed)
- [ ] Other (describe): ___

## Notes

- Risk assessment or manual verification steps (if any): ___
- Follow-up tickets or TODOs: ___
