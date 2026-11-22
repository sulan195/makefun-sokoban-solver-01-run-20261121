// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy: the generator builds each level as a hallway with 1-4 rooms
// hanging off it (one door each, 3-4 boxes per room), and a level's
// push-optimal count P* is the sum of its rooms' optima. So each room is
// solved independently with a push-optimal A* search, and the room
// solutions are stitched together with plain walks through the hallway.
// Pushes are what the score counts; walks are free.
//
// The A* is over pushes, not steps: a state is the sorted box squares plus
// the player's reachable region (canonicalised to its smallest square).
// Heuristic: exact min-cost assignment of boxes to goals with BFS
// distances on the room floor (admissible, so the search is optimal).
// Pruning: dead squares (reverse pull-reachability from the goals) and
// 2x2 box blocks with no goal inside. All budgets are counts of work
// (expansions), never the wall clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];

/** Expansions allowed for one room's optimal search, then fallbacks. */
const ROOM_OPTIMAL_CAP = 250_000;
const ROOM_WEIGHTED_CAP = 500_000;
const ROOM_GREEDY_CAP = 500_000;
/** Expansions allowed for the whole-level fallback search. */
const WHOLE_CAP = 2_000_000;

type Board = {
  w: number;
  h: number;
  /** 1 where a square is a wall. */
  wall: Uint8Array;
  /** 1 where a square is floor (not a wall). */
  floor: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function step(w: number, h: number, i: number, d: number): number {
  const x = (i % w) + DX[d]!;
  const y = Math.floor(i / w) + DY[d]!;
  if (x < 0 || x >= w || y < 0 || y >= h) return -1;
  return y * w + x;
}

function parse(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const floor = new Uint8Array(size);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      floor[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, floor, goals, boxes, player };
}

/** BFS distances from `from` over squares where `mask` is 1. */
function bfsDist(w: number, h: number, mask: Uint8Array, from: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  const q = new Int32Array(w * h);
  let qh = 0;
  let qt = 0;
  dist[from] = 0;
  q[qt++] = from;
  while (qh < qt) {
    const cur = q[qh++]!;
    for (let d = 0; d < 4; d++) {
      const n = step(w, h, cur, d);
      if (n < 0 || !mask[n] || dist[n] >= 0) continue;
      dist[n] = dist[cur]! + 1;
      q[qt++] = n;
    }
  }
  return dist;
}

/**
 * Flood fill from `from` over `walk` squares avoiding `boxesMask`.
 * Returns the reachable set and its smallest square (the canonical
 * player-region id: two states with the same boxes and the same reachable
 * region are the same state).
 */
function flood(
  w: number,
  h: number,
  walk: Uint8Array,
  boxesMask: Uint8Array,
  from: number,
): { reach: Uint8Array; min: number } {
  const reach = new Uint8Array(w * h);
  const q = new Int32Array(w * h);
  let qh = 0;
  let qt = 0;
  let min = from;
  reach[from] = 1;
  q[qt++] = from;
  while (qh < qt) {
    const cur = q[qh++]!;
    if (cur < min) min = cur;
    for (let d = 0; d < 4; d++) {
      const n = step(w, h, cur, d);
      if (n < 0 || !walk[n] || boxesMask[n] || reach[n]) continue;
      reach[n] = 1;
      q[qt++] = n;
    }
  }
  return { reach, min };
}

/**
 * Squares a box can sit on and still be pushed onto a goal, by reverse
 * pull-reachability from the goals (ignoring other boxes and whether the
 * player can reach the stand square, so this is a safe over-approximation:
 * a square it calls dead is truly dead).
 */
function liveSquares(
  w: number,
  h: number,
  boxMask: Uint8Array,
  walkMask: Uint8Array,
  goals: number[],
): Uint8Array {
  const live = new Uint8Array(w * h);
  const q = new Int32Array(w * h);
  let qh = 0;
  let qt = 0;
  for (const g of goals) {
    if (boxMask[g] && !live[g]) {
      live[g] = 1;
      q[qt++] = g;
    }
  }
  while (qh < qt) {
    const cur = q[qh++]!;
    for (let d = 0; d < 4; d++) {
      // A box at cur could have been pushed here from p, player at stand.
      const p = step(w, h, cur, d ^ 1);
      if (p < 0 || !boxMask[p] || live[p]) continue;
      const stand = step(w, h, p, d ^ 1);
      if (stand < 0 || !walkMask[stand]) continue;
      live[p] = 1;
      q[qt++] = p;
    }
  }
  return live;
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

function maskOf(w: number, h: number, boxes: number[]): Uint8Array {
  const m = new Uint8Array(w * h);
  for (const b of boxes) m[b] = 1;
  return m;
}

type Push = { from: number; to: number; dir: number };

type SearchOpts = {
  w: number;
  h: number;
  /** Squares a box may occupy. */
  boxMask: Uint8Array;
  /** Squares the player may walk on. */
  walkMask: Uint8Array;
  goals: number[];
  boxes0: number[];
  player0: number;
  /** Max states expanded (work budget, deterministic). */
  cap: number;
  /** 1 = optimal A*; higher = weighted (suboptimal, faster). */
  weight: number;
};

/**
 * Push-based A*. Returns the pushes from the start state to a solved
 * state, or null when the cap is hit or no solution exists.
 */
function solvePush(o: SearchOpts): Push[] | null {
  const { w, h, boxMask, walkMask, goals } = o;
  const n = o.boxes0.length;
  const gCount = goals.length;
  // More boxes than goals can never be solved: two boxes cannot share one
  // goal square.
  if (n > gCount) return null;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;

  // BFS distances from each goal over box squares (for the heuristic).
  const distFromGoal = goals.map((g) => bfsDist(w, h, boxMask, g));
  const live = liveSquares(w, h, boxMask, walkMask, goals);

  const hCache = new Map<string, number>();
  const heuristic = (boxes: number[]): number => {
    const key = String.fromCharCode(...boxes);
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let value: number;
    if (gCount <= 6) {
      // Exact min-cost assignment by DP over goal subsets: dp[mask] is the
      // cheapest cost of placing the first popcount(mask) boxes onto the
      // goals in mask.
      const dp = new Float64Array(1 << gCount).fill(Infinity);
      dp[0] = 0;
      for (let mask = 0; mask < 1 << gCount; mask++) {
        const k = popcount(mask);
        if (k >= n || dp[mask] === Infinity) continue;
        for (let j = 0; j < gCount; j++) {
          if (mask & (1 << j)) continue;
          const d = distFromGoal[j]![boxes[k]!]!;
          if (d < 0) continue;
          const nm = mask | (1 << j);
          const v = dp[mask]! + d;
          if (v < dp[nm]!) dp[nm] = v;
        }
      }
      value = Infinity;
      for (let mask = 0; mask < 1 << gCount; mask++) {
        if (popcount(mask) === n && dp[mask]! < value) value = dp[mask]!;
      }
    } else {
      // Admissible fallback: each box to its nearest goal.
      value = 0;
      for (const b of boxes) {
        let best = Infinity;
        for (let j = 0; j < gCount; j++) {
          const d = distFromGoal[j]![b]!;
          if (d >= 0 && d < best) best = d;
        }
        if (best === Infinity) {
          value = Infinity;
          break;
        }
        value += best;
      }
    }
    hCache.set(key, value);
    return value;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);
  const deadlocked = (boxes: number[], bm: Uint8Array): boolean => {
    for (const b of boxes) if (!live[b]) return true;
    // 2x2 block of boxes with no goal inside.
    for (const b of boxes) {
      const x = b % w;
      if (x + 1 >= w) continue;
      const b2 = b + 1;
      const b3 = b + w;
      const b4 = b + w + 1;
      if (b4 >= w * h) continue;
      if (bm[b2] && bm[b3] && bm[b4]) {
        if (!goalSet[b] && !goalSet[b2] && !goalSet[b3] && !goalSet[b4]) return true;
      }
    }
    return false;
  };

  // State storage (parallel arrays).
  const stBoxes: number[][] = [];
  const stPlayer: number[] = [];
  const stG: number[] = [];
  const stH: number[] = [];
  const stF: number[] = [];
  const stCnt: number[] = [];
  const stParent: number[] = [];
  const stPush: (Push | null)[] = [];
  const closed = new Map<string, number>();
  let counter = 0;

  const heap: number[] = [];
  const heapLess = (a: number, b: number): boolean =>
    stF[a]! < stF[b]! ||
    (stF[a]! === stF[b]! && (stH[a]! < stH[b]! || (stH[a]! === stH[b]! && stCnt[a]! < stCnt[b]!)));
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const stateKey = (boxes: number[], regionMin: number): string =>
    String.fromCharCode(...boxes, regionMin);

  const addState = (
    boxes: number[],
    player: number,
    g: number,
    parent: number,
    push: Push | null,
  ): number => {
    const bm = maskOf(w, h, boxes);
    const { min } = flood(w, h, walkMask, bm, player);
    const h0 = heuristic(boxes);
    const s = stBoxes.length;
    stBoxes.push(boxes);
    stPlayer.push(player);
    stG.push(g);
    stH.push(h0);
    stF.push(g + o.weight * h0);
    stCnt.push(counter++);
    stParent.push(parent);
    stPush.push(push);
    closed.set(stateKey(boxes, min), s);
    heapPush(s);
    return s;
  };

  const startBoxes = [...o.boxes0].sort((a, b) => a - b);
  if (solved(startBoxes)) return [];
  if (deadlocked(startBoxes, maskOf(w, h, startBoxes))) return null;
  addState(startBoxes, o.player0, 0, -1, null);

  let expanded = 0;
  while (heap.length > 0) {
    const s = heapPop();
    const boxes = stBoxes[s]!;
    const bm = maskOf(w, h, boxes);
    const { min } = flood(w, h, walkMask, bm, stPlayer[s]!);
    if (closed.get(stateKey(boxes, min)) !== s) continue; // stale entry
    if (solved(boxes)) {
      const pushes: Push[] = [];
      for (let t = s; stPush[t] !== null; t = stParent[t]!) pushes.push(stPush[t]!);
      pushes.reverse();
      return pushes;
    }
    if (++expanded > o.cap) return null;
    const { reach } = flood(w, h, walkMask, bm, stPlayer[s]!);
    for (let i = 0; i < n; i++) {
      const b = boxes[i]!;
      for (let d = 0; d < 4; d++) {
        const dest = step(w, h, b, d);
        if (dest < 0 || !boxMask[dest] || bm[dest]) continue;
        const behind = step(w, h, b, d ^ 1);
        if (behind < 0 || !walkMask[behind] || !reach[behind]) continue;
        const nb = boxes.slice();
        nb[i] = dest;
        nb.sort((a, b) => a - b);
        const nbMask = maskOf(w, h, nb);
        if (deadlocked(nb, nbMask)) continue;
        const h0 = heuristic(nb);
        if (!Number.isFinite(h0)) continue;
        const { min: nmin } = flood(w, h, walkMask, nbMask, b);
        const key = stateKey(nb, nmin);
        const prev = closed.get(key);
        if (prev !== undefined && stG[prev]! <= stG[s]! + 1) continue;
        addState(nb, b, stG[s]! + 1, s, { from: b, to: dest, dir: d });
      }
    }
  }
  return null;
}

/** Shortest walk (letters) from `from` to `to` avoiding `boxesMask`. */
function walkPath(
  w: number,
  h: number,
  walk: Uint8Array,
  boxesMask: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevDir = new Int8Array(w * h);
  const q = new Int32Array(w * h);
  let qh = 0;
  let qt = 0;
  prev[from] = -1;
  q[qt++] = from;
  while (qh < qt) {
    const cur = q[qh++]!;
    for (let d = 0; d < 4; d++) {
      const n = step(w, h, cur, d);
      if (n < 0 || !walk[n] || boxesMask[n] || prev[n] !== -2) continue;
      prev[n] = cur;
      prevDir[n] = d;
      if (n === to) {
        let out = '';
        for (let t = n; t !== from; t = prev[t]!) out = LETTER[prevDir[t]!]! + out;
        return out;
      }
      q[qt++] = n;
    }
  }
  return null;
}

type Room = {
  /** Squares of the room (door square included). */
  region: Uint8Array;
  regionList: number[];
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
};

/**
 * Split a level into rooms. The hallway is the row the player starts on;
 * each door is a floor square in a neighbouring row that touches hallway
 * floor; a room is the flood fill from its door without crossing the
 * hallway row. Returns null when the split does not cover every box and
 * goal exactly once.
 */
function decompose(board: Board): Room[] | null {
  const { w, h, floor } = board;
  const hy = Math.floor(board.player / w);
  const rooms: Room[] = [];
  const seen = new Uint8Array(w * h);
  for (const dy of [-1, 1]) {
    const r = hy + dy;
    if (r < 0 || r >= h) continue;
    for (let x = 0; x < w; x++) {
      const door = r * w + x;
      const entry = hy * w + x;
      if (!floor[door] || !floor[entry]) continue;
      if (seen[door]) continue; // wide opening: already part of a found room
      // Flood fill from the door, never crossing the hallway row.
      const region = new Uint8Array(w * h);
      const q = new Int32Array(w * h);
      let qh = 0;
      let qt = 0;
      region[door] = 1;
      q[qt++] = door;
      while (qh < qt) {
        const cur = q[qh++]!;
        for (let d = 0; d < 4; d++) {
          const n = step(w, h, cur, d);
          if (n < 0 || !floor[n] || region[n] || Math.floor(n / w) === hy) continue;
          region[n] = 1;
          q[qt++] = n;
        }
      }
      const regionList: number[] = [];
      const boxes: number[] = [];
      const goals: number[] = [];
      for (let i = 0; i < w * h; i++) {
        if (!region[i]) continue;
        regionList.push(i);
        if (board.boxes.includes(i)) boxes.push(i);
        if (board.goals.includes(i)) goals.push(i);
      }
      if (boxes.length === 0 || boxes.length !== goals.length) return null;
      for (const i of regionList) {
        if (seen[i]) return null; // rooms must not overlap
        seen[i] = 1;
      }
      rooms.push({ region, regionList, boxes, goals, door, entry });
    }
  }
  if (rooms.length === 0) return null;
  // Every box and goal must belong to exactly one room.
  for (const b of board.boxes) if (!seen[b]) return null;
  for (const g of board.goals) if (!seen[g]) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

/** Turn a push list into a move string, starting from `player`. */
function pushesToMoves(
  w: number,
  h: number,
  walkMask: Uint8Array,
  boxes0: number[],
  player0: number,
  pushes: Push[],
): string | null {
  let moves = '';
  let player = player0;
  let boxes = [...boxes0].sort((a, b) => a - b);
  for (const push of pushes) {
    const behind = step(w, h, push.from, push.dir ^ 1);
    const path = walkPath(w, h, walkMask, maskOf(w, h, boxes), player, behind);
    if (path === null) return null;
    moves += path;
    moves += LETTER[push.dir]!;
    player = push.from;
    boxes = boxes.map((b) => (b === push.from ? push.to : b));
    boxes.sort((a, b) => a - b);
  }
  return moves;
}

/**
 * Replay a move string on a board. Returns the final boxes/player, or
 * null on an illegal step. Mirrors the kit checker's replay exactly.
 */
function replay(
  board: Board,
  moves: string,
): { boxes: number[]; player: number; pushes: number } | null {
  const { w, h } = board;
  const boxAt = new Uint8Array(w * h);
  for (const b of board.boxes) boxAt[b] = 1;
  let player = board.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch);
    if (d < 0) return null;
    const next = step(w, h, player, d);
    if (next < 0 || board.wall[next]) return null;
    if (boxAt[next]) {
      const beyond = step(w, h, next, d);
      if (beyond < 0 || board.wall[beyond] || boxAt[beyond]) return null;
      boxAt[next] = 0;
      boxAt[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  const boxes: number[] = [];
  for (let i = 0; i < w * h; i++) if (boxAt[i]) boxes.push(i);
  return { boxes, player, pushes };
}

function solveLevel(text: string): string {
  const board = parse(text);
  const { w, h } = board;
  const goalSet = new Set(board.goals);

  // 1. Per-room solving.
  const rooms = decompose(board);
  if (rooms !== null) {
    let moves = '';
    let player = board.player;
    let boxes = [...board.boxes].sort((a, b) => a - b);
    let ok = true;
    for (const room of rooms) {
      const walkMask = new Uint8Array(w * h);
      for (const i of room.regionList) walkMask[i] = 1;
      walkMask[room.entry] = 1;
      const boxMask = room.region;
      let pushes: Push[] | null = solvePush({
        w,
        h,
        boxMask,
        walkMask,
        goals: room.goals,
        boxes0: room.boxes,
        player0: room.entry,
        cap: ROOM_OPTIMAL_CAP,
        weight: 1,
      });
      if (pushes === null) {
        pushes = solvePush({
          w,
          h,
          boxMask,
          walkMask,
          goals: room.goals,
          boxes0: room.boxes,
          player0: room.entry,
          cap: ROOM_WEIGHTED_CAP,
          weight: 2,
        });
      }
      if (pushes === null) {
        pushes = solvePush({
          w,
          h,
          boxMask,
          walkMask,
          goals: room.goals,
          boxes0: room.boxes,
          player0: room.entry,
          cap: ROOM_GREEDY_CAP,
          weight: 1000000,
        });
      }
      if (pushes === null) {
        ok = false;
        break;
      }
      // Walk from where we are to this room's entry, then the room moves.
      const walk = walkPath(w, h, board.floor, maskOf(w, h, boxes), player, room.entry);
      if (walk === null) {
        ok = false;
        break;
      }
      const roomMoves = pushesToMoves(w, h, walkMask, room.boxes, room.entry, pushes);
      if (roomMoves === null) {
        ok = false;
        break;
      }
      moves += walk + roomMoves;
      const after = replay({ ...board, boxes, player }, walk + roomMoves);
      if (after === null) {
        ok = false;
        break;
      }
      boxes = after.boxes;
      player = after.player;
    }
    if (ok) {
      const final = replay(board, moves);
      if (final !== null && final.boxes.every((b) => goalSet.has(b))) return moves;
    }
  }

  // 2. Whole-level fallback (same engine, no decomposition).
  let pushes: Push[] | null = solvePush({
    w,
    h,
    boxMask: board.floor,
    walkMask: board.floor,
    goals: board.goals,
    boxes0: board.boxes,
    player0: board.player,
    cap: WHOLE_CAP,
    weight: 1,
  });
  if (pushes === null) {
    pushes = solvePush({
      w,
      h,
      boxMask: board.floor,
      walkMask: board.floor,
      goals: board.goals,
      boxes0: board.boxes,
      player0: board.player,
      cap: WHOLE_CAP,
      weight: 3,
    });
  }
  if (pushes !== null) {
    const moves = pushesToMoves(w, h, board.floor, board.boxes, board.player, pushes);
    if (moves !== null) {
      const final = replay(board, moves);
      if (final !== null && final.boxes.every((b) => goalSet.has(b))) return moves;
    }
  }

  // 3. Give up on this level, but stay in the stated format.
  return '';
}

function solve(text: string): string {
  return solveLevel(text);
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
