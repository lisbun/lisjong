# RiichiLab request_action Adapter migration record

この文書はcurrent specificationではない。

lisjongで[Issue #38](https://github.com/lisbun/lisjong/issues/38) /
[Issue #39](https://github.com/lisbun/lisjong/issues/39)として実装した
RiichiLab protocol-facing decision bridge(`src/lisjong/riichilab_adapter/`:
`RiichiLabSeatAdapter` / `SendReadyResponse`、request_action parsing、MJAI
response正規化、possible_actions semantic validation、Adapter-specific error
hierarchy)は、
[`lisbun/lisjong-arena#27`](https://github.com/lisbun/lisjong-arena/issues/27) /
[PR #28](https://github.com/lisbun/lisjong-arena/pull/28)でArena-local
implementationへcanonical + physical migrationした。PR #28のactual merge commit
は`14cdd80cd3035c46c9d3f7bad034dda6c3b69f8c`である。

current protocol-facing decision bridgeの正本はArena側
[RiichiLab protocol-facing decision bridge](https://github.com/lisbun/lisjong-arena/blob/main/docs/riichilab-protocol-bridge.md)
とする。request_action protocol facts、MJAI response normalization、
possible_actions candidate semantics等のfull historical contractは、Arena側
current documentとlisjong側のこの記録との二重SoTを避けるため、本書には残さない。

lisjong [Issue #94](https://github.com/lisbun/lisjong/issues/94)では、Arena側
takeoverを確認したうえで、legacy `src/lisjong/riichilab_adapter/` package全体
(`adapter.py` / `request_action.py` / `mjai_response.py` /
`possible_action_validation.py` / `errors.py` / `__init__.py`)と対応する
protocol-facing legacy tests(`tests/test_riichilab_adapter.py` /
`tests/test_riichilab_request_action.py` / `tests/test_riichilab_mjai_response.py` /
`tests/test_riichilab_possible_action_validation.py`)を削除した。compatibility
wrapper / re-exportや`lisjong -> lisjong-arena`のreverse dependencyは残していない。

Issue #94 merge直後はArenaのlisjong dependency pinがcleanup前revisionを参照し得るため、
この時点をphysical duplicate完全解消とは扱わない。cleanup merge SHAをexact targetとする
Arena post-cleanup dependency pin syncが完了した時点で完全解消とする。

`build_decision()`(#23)、`execute_policy()`(#34)、`Policy` / `DecisionContext` /
`InternalAction`等のAI-side semanticsは、この移管の対象外でありlisjongに残る。
