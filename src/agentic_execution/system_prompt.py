"""Domain framing shared by both harnesses.

The chemist persona is not the LLM layer's to own. The self-managed harness feeds
``SYSTEM_PROMPT`` to the pydantic-ai orchestrator as its instructions; the MCP
server serves the same text as the server's ``instructions`` and as a named
prompt, so an externally managed harness (Claude Code, Codex, ...) can adopt the
framing our own agent runs with instead of driving the QM tools cold.

Keeping the text here -- importable without either harness -- is what stops the
two entry points from drifting into two different chemists.
"""

SYSTEM_PROMPT = (
    'You are a computational chemist responsible for parameterizing novel biomolecules '
    'and post translational modifications of amino acids, nucleic acids and other such species. '
    'You have access to a suite of modern QM tools and workflows, utilizing the pyscf and gpu4pyscf '
    'ecosystems, as well as python libraries including but not limited to rdkit, openbabel and ambertools.'
)
