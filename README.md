# lisjong

Personal Japanese riichi mahjong AI for RiichiEnv and RiichiLab.

> [!IMPORTANT]
> lisjong is an independent personal Japanese mahjong AI project developed by
> [lisbun](https://github.com/lisbun). It is not affiliated with any other
> project using the LisJong or lisjong name.

## 概要

lisjongは、日本式立直麻雀AIを自作し、ローカル対局からオンライン対局まで
同じAI Policyを再利用できる形で開発・評価するプロジェクトです。

現在は初期開発段階です。最初の到達目標は、学習済みmodelの強さではなく、
決定的な最小PolicyをRiichiEnvとRiichiLabへ安全に接続し、半荘を完走する
ことです。ロードマップと完了条件は
[親Issue](https://github.com/lisbun/lisjong/issues/1)で管理します。

## 位置づけ

| 対象 | 役割 |
| --- | --- |
| lisjong | 自作麻雀AI、Policy、学習・推論、接続Adapter、評価 |
| [RiichiEnv](https://riichi.dev/docs/local-testing) | ローカル対局・開発・回帰評価環境 |
| [RiichiLab](https://riichi.dev/) | オンライン接続先 |
| Mortal | 比較対象・互換性確認用の外部AI |
| [python-study](https://github.com/lisbun/python-study) | 将来の接続先となる自作麻雀基盤を含む学習repository |

Mortalやpython-studyのコード・modelをlisjongの内部実装として取り込むことは
初期目標に含めません。

## 開発方針

- 初期実装はPython 3.12を基準とする
- RiichiEnv、RiichiLabなど外部環境の型・protocolをAI Policyから分離する
- 各プレイヤーから観測可能な情報だけを判断へ使用する
- Policyは合法手からactionを選択し、外部送信前にも合法手を検証する
- 再現可能なseed、version、評価条件を記録する
- AIの強さより先に、接続の正しさ、半荘完走、テスト可能性を確立する
- Rustは先行導入せず、profilingで必要性が確認された処理に限って検討する

## ロードマップ

1. Python package、テスト、CI、文書の初期整備
2. RiichiEnvの実API・依存条件の調査
3. 共通Policy境界の設計
4. 学習modelを使わない最小Policyの実装
5. RiichiEnvで最初の局終了・半荘完走
6. RiichiLab validation・ランク戦1半荘
7. 接続MVP完了後に学習・評価を開始

Issue単位の現在地は
[GitHub Issues](https://github.com/lisbun/lisjong/issues)を正本とします。

## データ・model・秘密情報

このrepositoryには、次をcommitまたは再配布しません。

- RiichiLabのBot token、API key、その他の秘密情報
- 利用条件を確認していない牌譜・学習データ
- Mortalなど外部プロジェクトのmodel weight
- 大容量の生成model、raw data、実験artifact

外部データやmodelを利用する場合は、提供元、license、version、取得方法、
hashなどを確認し、repository本体とは分離して管理します。

## 開発状況

package構成、install方法、実行方法、テストコマンドは初期整備Issueで確定します。
現時点では、READMEに記載できる実行可能なAI実装はまだありません。

## License

lisjong自身のsource codeは[MIT License](LICENSE)で公開します。外部library、
model、牌譜、学習データには、それぞれの提供元のlicenseと利用条件が適用されます。
