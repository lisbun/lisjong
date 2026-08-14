"""RiichiLab request_action Adapter専用の例外。

Issue #38が担う責務境界(request_action入力validation、Observation
deserialize、seat-bound runtimeのseat照合、送信前possible_actions semantic
validation、MJAI response正規化)ごとに、呼び出し側が原因追跡できる最小限の
例外型だけを定義する。既存#23 (`riichienv_adapter`)・#34 (`policy_contract`)が
送出する例外はここで変更・再wrapせず、そのまま伝播させる。
"""


class RiichiLabAdapterError(Exception):
    """RiichiLab request_action Adapter境界のfail closed例外の基底class。"""


class MalformedRequestActionError(RiichiLabAdapterError):
    """parsed済みrequest_action相当dataが必須field欠落・型不正等でmalformedな場合。

    `type != "request_action"`、`request_id` / `possible_actions` /
    `observation`の欠落または安全に扱えない型を含む。
    """


class ObservationDeserializeError(RiichiLabAdapterError):
    """base64 observationを4-player `riichienv.Observation`へ復元できない場合。"""


class SeatMismatchError(RiichiLabAdapterError):
    """deserialize済みObservationのplayer_idが、このAdapterのbound seatと一致しない場合。"""


class PossibleActionsValidationError(RiichiLabAdapterError):
    """送信予定Actionが、server提示`possible_actions`へ一意にsemantic matchしない場合。

    0件一致、複数件一致(ambiguous)、比較不能のいずれもこの例外で拒否する。
    """


class ProtocolConversionError(RiichiLabAdapterError):
    """resolve済みRiichiEnv Actionを、RiichiLab Bot-to-Server response相当の
    MJAI dictへ変換できない場合。
    """
