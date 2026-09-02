"""model-facing action vocabulary package。

Issue #149「Learned Policy Stage 0 — InternalAction ↔ model-facing action
vocabularyとlegal maskの契約を確立する」で、learned Policyが固定長のaction出力を
扱えるように、fixed-sizeかつversionedなaction vocabulary、`InternalAction`との
codec、`DecisionContext.legal_actions`から導出するfixed-size legal maskを、
ML依存のないlisjong所有の契約として追加した。

```text
semantic identity
    = InternalAction dataclass value equality

model action index
    = versioned adapter representation
```

model action indexは新しいAction identityではない。合法性の根拠でも、
`legal_actions`のtuple indexでもない。`resolve_legal_action()`は常に同じdecisionの
canonical legal Action（`decision.legal_actions`側のobject）を返し、
`execute_policy()`のsignature、validation、例外semanticsは変更しない。

意味契約の正本はdocs/action-vocabulary.md、semantic identityの正本は
docs/action-identity.mdである。

このpackageは`lisjong.policy_contract`のvalue型とPython標準libraryだけへ依存し、
逆方向（`policy_contract -> action_vocabulary`）の依存は作らない。feature encoder、
tensor schema、HandBelief consumer、model architecture、trainingは本packageの
責務ではなく、後続Issueで扱う。
"""

from lisjong.action_vocabulary.action_codec import (
    ACTION_VOCABULARY_BLOCKS,
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    decode_action,
    encode_action,
)
from lisjong.action_vocabulary.errors import (
    ActionEncodingError,
    ActionIndexCollisionError,
    ActionIndexError,
    ActionVocabularyError,
    IllegalActionIndexError,
    UnsupportedActionVocabularyVersionError,
)
from lisjong.action_vocabulary.legal_mask import (
    build_legal_action_mask,
    encode_legal_actions,
    resolve_legal_action,
)

__all__ = [
    "ACTION_VOCABULARY_BLOCKS",
    "ACTION_VOCABULARY_SIZE",
    "ACTION_VOCABULARY_VERSION",
    "ActionEncodingError",
    "ActionIndexCollisionError",
    "ActionIndexError",
    "ActionVocabularyError",
    "IllegalActionIndexError",
    "UnsupportedActionVocabularyVersionError",
    "build_legal_action_mask",
    "decode_action",
    "encode_action",
    "encode_legal_actions",
    "resolve_legal_action",
]
