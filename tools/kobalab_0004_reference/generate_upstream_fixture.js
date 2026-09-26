/*
 * Development-only upstream differential fixture generator for lisjong #211.
 *
 * Runs the pinned upstream kobalab/majiang-ai legacy player 0004 on top of the
 * pinned @kobalab/majiang-core and records comparable intermediate values
 * (shanten, improving tiles, SuanPai remaining counts, paijia, per-candidate
 * ukeire) and final decisions in a language-neutral JSON fixture.
 *
 * This script is NOT a lisjong runtime dependency and is NOT run in CI.
 * Regenerate with:
 *
 *   cd tools/kobalab_0004_reference
 *   npm install
 *   node generate_upstream_fixture.js > ../../tests/fixtures/kobalab_0004_upstream.json
 *
 * Upstream: https://github.com/kobalab/majiang-ai (MIT License,
 * Copyright (c) Satoshi Kobayashi). Only the published modules are loaded;
 * no upstream source is copied into this file.
 */
"use strict";

const Majiang = require('@kobalab/majiang-core');
const Player0004 = require('@kobalab/majiang-ai/legacy/player-0004');
const SuanPai0004 = require('@kobalab/majiang-ai/legacy/suanpai-0004');

const MAJIANG_AI_COMMIT = 'e75a9720a12b84c03e6c61c3960c1844b8982eb4';
const MAJIANG_CORE_VERSION = require(
    '@kobalab/majiang-core/package.json').version;

/* ---------------------------------------------------------------- PRNG --- */

function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

/* ------------------------------------------------------------- helpers --- */

function concealedTiles(shoupai) {
    // Expand _bingpai into tile strings; red fives are "s0".
    const tiles = [];
    for (const s of ['m', 'p', 's', 'z']) {
        const b = shoupai._bingpai[s];
        for (let n = 1; n < b.length; n++) {
            let count = b[n];
            if (s != 'z' && n == 5) {
                for (let i = 0; i < b[0]; i++) tiles.push(s + '0');
                count -= b[0];
            }
            for (let i = 0; i < count; i++) tiles.push(s + n);
        }
    }
    return tiles;
}

function copyPaishu(suanpai) {
    const p = suanpai._paishu;
    return { m: [...p.m], p: [...p.p], s: [...p.s], z: [...p.z] };
}

function evaluateDapai(player) {
    // Mirrors the observable steps of player-0004 select_dapai() to expose
    // intermediate values; the final decision is taken from the upstream call.
    const shoupai = player.shoupai;
    const n_xiangting = Majiang.Util.xiangting(shoupai);
    const paijia = player._suanpai.make_paijia();
    const order = player.get_dapai(shoupai).reverse()
                        .sort((a, b) => paijia(a) - paijia(b));
    const candidates = order.map(p => {
        const after = shoupai.clone().dapai(p);
        const xiangting = Majiang.Util.xiangting(after);
        const tingpai = Majiang.Util.tingpai(after);
        const ev = tingpai.map(t => player._suanpai._paishu[t[0]][t[1]])
                          .reduce((x, y) => x + y, 0);
        return { p, paijia: paijia(p), xiangting, tingpai, ev };
    });
    return { n_xiangting, candidates };
}

function legalSummary(player) {
    const shoupai = player.shoupai;
    return {
        dapai: player.get_dapai(shoupai),
        lizhi: player.allow_lizhi(shoupai) || [],
        hule: !! player.allow_hule(shoupai, null, false),
        gang: player.get_gang_mianzi(shoupai) || [],
        pingju: !! player.allow_pingju(shoupai),
    };
}

function publicState(player) {
    const model = player._model;
    return {
        zhuangfeng: model.zhuangfeng,
        menfeng: player._menfeng,
        paishu: model.shan.paishu,
        baopai: [...model.shan.baopai],
        he: model.he.map(he => [...he._pai]),
        fulou: model.shoupai.map(s => [...s._fulou]),
        lizhi: model.shoupai.map(s => !! s.lizhi),
        concealed: concealedTiles(player.shoupai),
        zimo: player.shoupai._zimo,
    };
}

/* ----------------------------------------------------- game-derived part --- */

class RecordingPlayer extends Player0004 {

    constructor(recorder) {
        super();
        this._recorder = recorder;
    }

    action_zimo(zimo, gangzimo) {
        if (zimo.l != this._menfeng) return super.action_zimo(zimo, gangzimo);
        const record = this._recorder.shouldRecord();
        const state = record ? publicState(this) : null;
        const legal = record ? legalSummary(this) : null;
        const paishu = record ? copyPaishu(this._suanpai) : null;
        const evaluated = record ? evaluateDapai(this) : null;
        const callback = this._callback;
        this._callback = reply => {
            if (record) {
                this._recorder.add({
                    state, legal, suanpai_paishu: paishu,
                    evaluation: evaluated,
                    decision: reply || {},
                    gangzimo: !! gangzimo,
                });
            }
            callback(reply);
        };
        super.action_zimo(zimo, gangzimo);
    }
}

class Recorder {
    constructor(every, limit) {
        this.every = every; this.limit = limit;
        this.count = 0; this.records = [];
    }
    shouldRecord() {
        this.count++;
        return this.records.length < this.limit && this.count % this.every == 0;
    }
    add(r) { this.records.push(r); }
}

function runGame(seed, every, limit) {
    const saved = Math.random;
    Math.random = mulberry32(seed);
    try {
        const recorder = new Recorder(every, limit);
        const players = [0, 1, 2, 3].map(() => new RecordingPlayer(recorder));
        const game = new Majiang.Game(players, () => {}, Majiang.rule());
        game.do_sync();
        return recorder.records.map((r, i) => (
            Object.assign({ id: `game-${seed}-${i}` }, r)));
    }
    finally {
        Math.random = saved;
    }
}

/* ---------------------------------------------------------- crafted part --- */

function craftedPlayer(c) {
    // Injects a single-seat snapshot into the upstream player. Only the fields
    // read by player-0004 decision methods are provided.
    const rule = Majiang.rule();
    const player = new Player0004();
    const menfeng = c.menfeng ?? 0;
    const shoupai = Majiang.Shoupai.fromString(c.shoupai);
    const others = [0, 1, 2, 3].map(l => l == menfeng ? shoupai
                                               : new Majiang.Shoupai());
    player._rule = rule;
    player._id = menfeng;
    player._menfeng = menfeng;
    player._n_gang = c.n_gang ?? 0;
    player._diyizimo = c.diyizimo ?? false;
    player._neng_rong = true;
    player._model = {
        zhuangfeng: c.zhuangfeng ?? 0,
        lunban: c.lunban ?? menfeng,
        shan: { paishu: c.paishu ?? 60, baopai: c.baopai },
        shoupai: others,
        he: [0, 1, 2, 3].map(() => new Majiang.He()),
        defen: [25000, 25000, 25000, 25000],
    };
    const next = (menfeng + 1) % 4;
    for (const p of c.seen || []) player._model.he[next].dapai(p);

    const suanpai = new SuanPai0004(rule['赤牌']);
    suanpai._zhuangfeng = c.zhuangfeng ?? 0;
    suanpai._menfeng = menfeng;
    suanpai._baopai = [...c.baopai];
    for (const p of c.baopai) suanpai.decrease(p);
    for (const p of concealedTiles(shoupai)) suanpai.decrease(p);
    for (const p of c.seen || []) suanpai.decrease(p);
    player._suanpai = suanpai;
    return player;
}

function craftedTurn(c) {
    const player = craftedPlayer(c);
    let decision;
    player._callback = reply => { decision = reply || {}; };
    player.action_zimo({ l: player._menfeng, p: '' }, c.gangzimo ?? false);
    return {
        id: c.id,
        note: c.note,
        state: publicState(player),
        legal: legalSummary(player),
        suanpai_paishu: copyPaishu(player._suanpai),
        evaluation: player.shoupai._zimo ? evaluateDapai(player) : null,
        decision,
        gangzimo: c.gangzimo ?? false,
        diyizimo: c.diyizimo ?? false,
    };
}

function craftedChankan(c) {
    const player = craftedPlayer(c);
    const l = (player._menfeng + 1) % 4;
    player._model.lunban = l;
    return {
        id: c.id,
        note: c.note,
        state: publicState(player),
        gang: { l, m: c.gang },
        hule: !! player.select_hule({ l, m: c.gang }, true),
    };
}

const CRAFTED_TURNS = [
    { id: 'kan-keeps-shanten', note: 'ankan keeps shanten -> gang',
      shoupai: 'm1111p123s789z11p56', baopai: ['s1'] },
    { id: 'kan-worsens-shanten', note: 'ankan worsens shanten -> no gang',
      shoupai: 'm111123p456s789z56', baopai: ['s1'] },
    { id: 'kan-order', note: 'two ankan candidates, first in m->z order kept',
      shoupai: 'm1111p5555s234z112', baopai: ['s1'] },
    { id: 'kan-order-skip-first', note: 'first ankan worsens, second keeps',
      shoupai: 'm111123p5555z117s9', baopai: ['s1'] },
    { id: 'kyuushu-yes', note: 'nine terminal/honor types, shanten >= 4',
      shoupai: 'm124689p139s1z1234', baopai: ['s1'], diyizimo: true },
    { id: 'kyuushu-no-shanten3', note: 'kokushi shanten 3 -> no abort',
      shoupai: 'm12589p139s19z1234', baopai: ['s1'], diyizimo: true },
    { id: 'red-five-tie', note: 'red vs normal 5 complete tie',
      shoupai: 'm123456789p1s0555', baopai: ['z3'] },
    { id: 'honor-winds', note: 'round/seat wind and dragon paijia',
      shoupai: 'm123456789p1z1237', baopai: ['s1'], zhuangfeng: 1, menfeng: 2 },
    { id: 'dora-honor', note: 'honor dora squared weight',
      shoupai: 'm123456789p1z1234', baopai: ['z1'] },
    { id: 'edge-suited', note: 'edge ranks 1/9 isolated',
      shoupai: 'm123456789p11s19z1', baopai: ['z5'] },
    { id: 'dora-suited-neighbour', note: 'dora neighbours in paijia',
      shoupai: 'm123456789p11s28z1', baopai: ['s2'] },
    { id: 'visible-changes-ukeire', note: 'visible tiles reduce ukeire',
      shoupai: 'm12345678p1134s5z1', baopai: ['z1'],
      seen: ['m9', 'm9', 'm9', 's5', 's5'] },
    { id: 'chiitoi', note: 'chiitoitsu shanten and tingpai',
      shoupai: 'm1133p2255s4477z12', baopai: ['z1'] },
    { id: 'chiitoi-quad', note: 'four identical tiles do not form two pairs',
      shoupai: 'm1111p2255s4477z12', baopai: ['z1'] },
    { id: 'kokushi', note: 'kokushi-shaped hand',
      shoupai: 'm15569p19s19z12345', baopai: ['z7'] },
    { id: 'riichi-immediate', note: 'tenpai discard -> immediate riichi',
      shoupai: 'm123456789p11s24z1', baopai: ['z7'] },
    { id: 'tsumogiri-tie', note: 'tsumogiri vs tedashi of identical tile',
      shoupai: 'm123456789p11s11s1', baopai: ['z7'] },
    { id: 'red-tsumogiri', note: 'red five draw and identical normal five',
      shoupai: 'm123456789p11s5z1s0', baopai: ['z7'] },
    { id: 'after-kan-hand', note: 'discard after ankan (10+1 tiles)',
      shoupai: 'm234p456s67z11z5,m1111', baopai: ['z7', 's3'], gangzimo: true },
];

const CRAFTED_CHANKAN = [
    { id: 'chankan-ankan-reject', note: 'kokushi tenpai, ankan z1 -> no ron',
      shoupai: 'm19p19s19z2345677', baopai: ['z7'], gang: 'z1111' },
    { id: 'chankan-kakan-accept', note: 'ron on kakan tile allowed',
      shoupai: 'm123456789p11s46', baopai: ['z7'], gang: 's555=5' },
];

/* --------------------------------------------------------- shanten part --- */

function randomHands(seed, n) {
    const random = mulberry32(seed);
    const wall = [];
    for (const s of ['m', 'p', 's', 'z'])
        for (let k = 1; k <= (s == 'z' ? 7 : 9); k++)
            for (let i = 0; i < 4; i++)
                wall.push(s + (s != 'z' && k == 5 && i == 0 ? 0 : k));
    const hands = [];
    for (let i = 0; i < n; i++) {
        const w = [...wall];
        for (let j = w.length - 1; j > 0; j--) {
            const k = Math.floor(random() * (j + 1));
            [w[j], w[k]] = [w[k], w[j]];
        }
        // bias some hands to be close to complete by drawing mostly one suit
        const size = [13, 14, 10, 11][i % 4];
        let picked;
        if (i % 3 == 0) {
            const suit = 'mps'[i % 9 % 3];
            picked = w.filter(p => p[0] == suit || p[0] == 'z')
                      .slice(0, size);
        }
        else picked = w.slice(0, size);
        hands.push(picked);
    }
    return hands;
}

function shantenCase(tiles) {
    const s = { m: '', p: '', s: '', z: '' };
    for (const p of tiles) s[p[0]] += p[1];
    let str = ['m', 'p', 's', 'z'].filter(k => s[k]).map(k => k + s[k]).join('');
    // Hands of 10/11 tiles carry one ankan so majiang-core sees a fixed meld.
    if (tiles.length <= 11) str += ',z7777';
    const shoupai = Majiang.Shoupai.fromString(str);
    // Remove the placeholder ankan tiles from counts if they collide.
    const xiangting = Majiang.Util.xiangting(shoupai);
    const tingpai = tiles.length % 3 == 1 ? Majiang.Util.tingpai(shoupai) : null;
    return { tiles, xiangting, tingpai };
}

function shantenCases() {
    return randomHands(211, 240)
        .filter(tiles => tiles.length > 11
                         || tiles.filter(p => p == 'z7').length == 0)
        .map(shantenCase);
}

/* ------------------------------------------------------------------ main --- */

const output = {
    generator: 'tools/kobalab_0004_reference/generate_upstream_fixture.js',
    upstream: {
        majiang_ai: {
            repository: 'https://github.com/kobalab/majiang-ai',
            commit: MAJIANG_AI_COMMIT,
            files: ['legacy/player-0004.js', 'legacy/suanpai-0004.js'],
            license: 'MIT',
        },
        majiang_core: { package: '@kobalab/majiang-core',
                        version: MAJIANG_CORE_VERSION },
        rule: 'Majiang.rule() defaults',
    },
    games: [ ...runGame(20260926, 11, 45), ...runGame(211, 13, 45) ],
    crafted_turns: CRAFTED_TURNS.map(craftedTurn),
    crafted_chankan: CRAFTED_CHANKAN.map(craftedChankan),
    shanten: shantenCases(),
};

process.stdout.write(JSON.stringify(output, null, 0)
    .replace(/\},\{"id"/g, '},\n{"id"')
    .replace(/\},\{"tiles"/g, '},\n{"tiles"') + '\n');
