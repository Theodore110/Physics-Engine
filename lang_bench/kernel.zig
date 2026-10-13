const Body = struct { x: f32, y: f32, vx: f32, vy: f32, inv_m: f32, r: f32 };
const Pair = struct { a: u32, b: u32 };
const Contact = struct { a: u32, b: u32, nx: f32, ny: f32, bias: f32, jn: f32 };
const MAXN = 400;
var bs: [MAXN * MAXN]Body = undefined;
var ps: [MAXN * MAXN * 4]Pair = undefined;
var cs: [MAXN * MAXN * 4]Contact = undefined;
var st: u32 = 12345;
fn rnd() f32 {
    st = st *% 1664525 +% 1013904223;
    return @as(f32, @floatFromInt(st >> 8)) / 16777216.0;
}
export fn k_run(n: c_int, steps: c_int) f32 {
    st = 12345;
    const dt: f32 = 1.0 / 60.0;
    const g: f32 = -9.8;
    const beta: f32 = 0.2;
    const slop: f32 = 0.01;
    const nn: usize = @intCast(n);
    var j: usize = 0;
    while (j < nn) : (j += 1) {
        var i: usize = 0;
        while (i < nn) : (i += 1) {
            const rx = rnd();
            const ry = rnd();
            const rv = rnd();
            bs[j * nn + i] = .{
                .x = @as(f32, @floatFromInt(i)) + (rx - 0.5) * 0.1,
                .y = @as(f32, @floatFromInt(j)) + (ry - 0.5) * 0.1,
                .vx = if (j == 0) 0.0 else (rv - 0.5) * 0.2,
                .vy = 0.0,
                .inv_m = if (j == 0) 0.0 else 1.0,
                .r = 0.55,
            };
        }
    }
    var np: usize = 0;
    const di = [4]i32{ 1, 0, 1, -1 };
    const dj = [4]i32{ 0, 1, 1, 1 };
    var jj0: i32 = 0;
    while (jj0 < n) : (jj0 += 1) {
        var ii0: i32 = 0;
        while (ii0 < n) : (ii0 += 1) {
            for (0..4) |k| {
                const ii = ii0 + di[k];
                const jj = jj0 + dj[k];
                if (ii < 0 or ii >= n or jj >= n) continue;
                if (jj0 == 0 and jj == 0) continue;
                ps[np] = .{ .a = @intCast(jj0 * n + ii0), .b = @intCast(jj * n + ii) };
                np += 1;
            }
        }
    }
    const nb = nn * nn;
    var s: c_int = 0;
    while (s < steps) : (s += 1) {
        for (bs[0..nb]) |*b| {
            if (b.inv_m > 0.0) b.vy += g * dt;
        }
        var nc: usize = 0;
        for (ps[0..np]) |p| {
            const a = bs[p.a];
            const b = bs[p.b];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const d2 = dx * dx + dy * dy;
            const rr = a.r + b.r;
            if (d2 < rr * rr and d2 > 1e-12) {
                const d = @sqrt(d2);
                const inv = 1.0 / d;
                var pen = rr - d - slop;
                if (pen < 0.0) pen = 0.0;
                cs[nc] = .{ .a = p.a, .b = p.b, .nx = dx * inv, .ny = dy * inv, .bias = (beta / dt) * pen, .jn = 0.0 };
                nc += 1;
            }
        }
        var it: usize = 0;
        while (it < 10) : (it += 1) {
            for (cs[0..nc]) |*c| {
                const A = &bs[c.a];
                const B = &bs[c.b];
                const ia = A.inv_m;
                const ib = B.inv_m;
                const vn = (B.vx - A.vx) * c.nx + (B.vy - A.vy) * c.ny;
                var lam = (c.bias - vn) / (ia + ib);
                var nj = c.jn + lam;
                if (nj < 0.0) nj = 0.0;
                lam = nj - c.jn;
                c.jn = nj;
                const px = lam * c.nx;
                const py = lam * c.ny;
                A.vx -= px * ia;
                A.vy -= py * ia;
                B.vx += px * ib;
                B.vy += py * ib;
            }
        }
        for (bs[0..nb]) |*b| {
            b.x += b.vx * dt;
            b.y += b.vy * dt;
        }
    }
    var sum: f64 = 0;
    for (bs[0..nb]) |b| sum += @as(f64, b.x) + @as(f64, b.y);
    return @floatCast(sum);
}
