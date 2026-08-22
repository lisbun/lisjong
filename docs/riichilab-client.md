# RiichiLab lower-level runtime migration record

この文書はcurrent runtime specificationではない。

lisjongでIssues #39 / #42 / #45として実装したRiichiLab lower-level runtime
(client errors / Session / Transport / protocol trace writer)は、
[`lisbun/lisjong-arena#23`](https://github.com/lisbun/lisjong-arena/issues/23) /
[PR #24](https://github.com/lisbun/lisjong-arena/pull/24)でArena-local implementationへ
canonical + physical migrationした。PR #24のmerge commitは
`9628271289a623fee30712409b5fd19585761625`である。

current runtime contractの正本はArena側
[RiichiLab client runtime contract](https://github.com/lisbun/lisjong-arena/blob/main/docs/riichilab-client.md)
とする。Adapter固有のrequest_action parsing、Observation / Policy projection、
MJAI response変換、`possible_actions` semantic validationの正本は、引き続きlisjong側
[RiichiLab request_action Adapter](riichilab-adapter.md)である。

lisjong Issue #91ではlegacy `src/lisjong/riichilab_client/` packageを削除し、
compatibility wrapper / re-exportや`lisjong -> lisjong-arena`のreverse dependencyを
残さない。

Issue #91 merge直後はArenaのlisjong dependency pinがcleanup前revisionを参照し得るため、
この時点をphysical duplicate完全解消とは扱わない。cleanup merge SHAをexact targetとする
Arena post-cleanup dependency pin syncが完了した時点で完全解消とする。
