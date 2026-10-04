// Injected into the T-Rex runner page by examples/dino. Samples the game state
// HZ times per second, asks the local local server for an action, and presses
// keys through the game's own handlers (the same path as a real keyboard).
(function () {
  'use strict';
  var HZ = __HZ__;
  var JUMP = 32, DUCK = 40;
  var CLIENT = Math.random().toString(36).slice(2, 6);
  var busy = false, down = {}, games = 0, best = 0, stats = { n: 0, ms: 0 }, rtt = null;

  function key(type, code) {
    var r = Runner.instance_;
    var e = { type: type, keyCode: code, preventDefault: function () {}, currentTarget: null, button: 0 };
    if (type === 'keydown') { down[code] = true; r.onKeyDown(e); } else { down[code] = false; r.onKeyUp(e); }
  }

  var hud = document.createElement('pre');
  hud.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);top:240px;margin:0;padding:10px 14px;font:13px/1.45 ui-monospace,monospace;' +
    'background:#202124;color:#e8eaed;border-radius:8px;min-width:340px;z-index:10';
  document.body.appendChild(hud);

  function state(r) {
    var t = r.tRex;
    return {
      client: CLIENT, sample_ms: 1000 / HZ, rtt_ms: rtt === null ? 22.5 : rtt, playing: r.playing, crashed: r.crashed, speed: r.currentSpeed, distance: Math.ceil(r.distanceRan),
      trex: { x: t.xPos, y: t.yPos, ground_y: t.groundYPos, width: t.ducking ? t.config.WIDTH_DUCK : t.config.WIDTH,
              jumping: t.jumping, ducking: t.ducking },
      obstacles: r.horizon.obstacles.map(function (o) {
        return { type: o.typeConfig.type, x: o.xPos, y: o.yPos, width: o.width, height: o.typeConfig.height };
      })
    };
  }

  function apply(action, r) {
    var t = r.tRex;
    if (action !== 'duck' && down[DUCK]) key('keyup', DUCK);
    if (action === 'jump' && !t.jumping && !t.ducking) key('keydown', JUMP);
    if (action === 'duck' && !down[DUCK]) key('keydown', DUCK);
  }

  function tick() {
    var r = Runner.instance_;
    if (!r || !r.tRex) return;
    if (down[JUMP] && !r.tRex.jumping) key('keyup', JUMP); // full-height jumps: release on landing
    if (r.crashed) {
      if (!r._decideCounted) {
        var score = r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan));
        r._decideCounted = true; games++; best = Math.max(best, score);
        fetch('/gameover', { method: 'POST', body: JSON.stringify({ client: CLIENT, score: score }) });
        setTimeout(function () { r._decideCounted = false; down = {}; r.restart(); }, 1500);
      }
      return;
    }
    // The game pauses itself when the window loses focus; a spectator demo resumes it.
    if (r.paused && !r.crashed) {
      down = {}; r.tRex.reset();
      if (!r.activated) r.tRex.jumpCount = 1; // paused before the intro: let it start
      r.play(); return;
    }
    if (!r.playing) { key('keydown', JUMP); return; } // start the first game
    if (busy) return; // one request in flight; skip this sample
    busy = true;
    var t0 = performance.now();
    fetch('/act', { method: 'POST', body: JSON.stringify(state(r)) })
      .then(function (res) { return res.json(); })
      .then(function (d) {
        busy = false;
        var dt = performance.now() - t0;
        stats.n++; stats.ms += dt;
        rtt = rtt === null ? dt : 0.8 * rtt + 0.2 * dt; // smoothed round trip, sent with the next state
        if (!Runner.instance_.crashed) apply(d.action, Runner.instance_);
        var p = Object.keys(d.probabilities).map(function (k) { return k + ' ' + (100 * d.probabilities[k]).toFixed(0) + '%'; });
        hud.textContent = 'GLiNER2.5-Decide (' + d.device + ') plays T-Rex   [viewer ' + CLIENT + ']\n' +
          'action   ' + d.action.toUpperCase() + '   [' + p.join(', ') + ']\n' +
          'obstacle ' + d.summary + '\n' +
          'model    ' + d.model_ms.toFixed(1) + ' ms   round trip ' + (stats.ms / stats.n).toFixed(1) + ' ms   ' + HZ + ' Hz\n' +
          'games    ' + games + '   best ' + best + '   now ' + r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan));
      })
      .catch(function () { busy = false; });
  }

  window.addEventListener('load', function () { setInterval(tick, 1000 / HZ); });
})();
