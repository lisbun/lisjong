"""model-facing action vocabularyのfail-closed例外階層。

docs/action-vocabulary.md「Fail closed」の意味契約を実装する。

codecは、semantic distinctionを失う変換、同一decision内で一意に解決できない
index、未対応のvocabulary versionを、任意のfallback Actionへ置換しない。
いずれの失敗も例外としてcallerへ伝播させる。

`lisjong.policy_contract`の`PolicyActionValidationError`とは別の階層とする。
model-facing indexはsemantic identityでもlegal action validationの根拠でもなく、
`execute_policy()`のvalidation semanticsを変更しないためである。
"""


class ActionVocabularyError(Exception):
    """model-facing action vocabulary contractのfail-closed基底例外。"""


class UnsupportedActionVocabularyVersionError(ActionVocabularyError):
    """要求されたvocabulary versionをこの実装が提供しない場合。"""


class ActionEncodingError(ActionVocabularyError):
    """InternalActionをmodel action indexへencodeできない場合。"""


class ActionIndexError(ActionVocabularyError):
    """indexがvocabulary上のindexとして成立しない場合。"""


class IllegalActionIndexError(ActionVocabularyError):
    """vocabulary上は有効だが、当該decisionでlegalでないindexの場合。"""


class ActionIndexCollisionError(ActionVocabularyError):
    """同一decisionの複数legal actionsが同じindexへ衝突した場合。"""
