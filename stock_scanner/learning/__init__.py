"""Research-only analysis tools (pattern mining, and later hypothesis
generation). Nothing in stock_scanner/pipeline/ imports this package, and
nothing here writes to signals/model_registry/scanner_config.yaml — this
boundary is deliberate, not incidental: it is what makes "cannot affect
production behavior" a structural property instead of a convention to
remember. See docs/LEARNING_AGENT_ARCHITECTURE.md.
"""
