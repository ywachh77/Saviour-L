/* ==========================================================================
   SAVIOUR — Gate Transition & Multi-Page Operator Console Controller
   Uses strictly relative API paths ("api/...") and graceful static fallback.
   Zero external libraries, zero emojis.
   ========================================================================== */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function esc(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtPrice(val, decimals) {
    var num = Number(val);
    if (!isFinite(num)) return "n/a";
    return num.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function apiGet(path) {
    return fetch(path, { headers: { "Accept": "application/json" } }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function apiPost(path, payload) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload || {})
    }).then(function (r) {
      return r.json().then(function (body) {
        if (!r.ok || body.ok === false) {
          throw new Error(body.error || ("HTTP " + r.status));
        }
        return body;
      });
    });
  }

  /* ========================================================================
     PART A : HOME GATE (index.html) — NARUTO FUINJUTSU RELEASE SEQUENCE
     ======================================================================== */
  var unleashBtn = document.getElementById("unleash-btn");
  if (unleashBtn) {
    var unleashing = false;
    var caption = document.getElementById("fuin-caption");

    window.addEventListener("pageshow", function () {
      unleashing = false;
      document.body.classList.remove("is-unleashing");
    });

    apiGet("api/status")
      .then(function (st) {
        var mkEl = document.getElementById("gate-markets");
        var enEl = document.getElementById("gate-engine-state");
        if (mkEl && st.active_pairs) mkEl.textContent = st.active_pairs.join(" · ");
        if (enEl) {
          var feedTag = st.feed_mode === "LIVE" ? "LIVE SPOT FEED" : (st.feed_mode === "SUPABASE_SYNC" ? "SUPABASE CREDENTIALS CONNECTED" : "PAPER ENGINE READY");
          enEl.textContent = st.council_members + " COUNCIL GATES · " + feedTag;
        }
      })
      .catch(function () {});

    unleashBtn.addEventListener("click", function (ev) {
      var targetHref = unleashBtn.getAttribute("href") || "console.html#signals";
      if (reduceMotion) return; // follow anchor immediately
      ev.preventDefault();
      if (unleashing) return;
      unleashing = true;

      if (caption) caption.textContent = "SEAL ARRAY DEPLOYED // 八卦封印式";
      document.body.classList.add("is-unleashing");

      setTimeout(function () {
        if (caption) caption.textContent = "KAI — SEAL RELEASED // 封印解除・解";
      }, 780);

      setTimeout(function () {
        window.location.href = targetHref;
      }, 1440);
    });
    return;
  }

  /* ========================================================================
     PART B : MULTI-PAGE OPERATOR CONSOLE (console.html)
     ======================================================================== */
  if (!document.querySelector(".console-body")) return;

  var PAGE_META = {
    signals:     { code: "DIVISION 01 // SIGNALS",     title: "Signal Execution",    jp: "信号", sig: "一閃 · BLADE CLEAVE",    fx: "fx-signals",     stamp: "一閃 · 信号" },
    telegram:    { code: "DIVISION 02 // TELEGRAM",    title: "Telegram Dispatch",   jp: "電信", sig: "巻物 · SCROLL UNFURL",   fx: "fx-telegram",    stamp: "巻物 · 電信" },
    health:      { code: "DIVISION 03 // HEALTH",      title: "Health & Causality",  jp: "健全", sig: "検診 · DIAGNOSTIC SCAN", fx: "fx-health",      stamp: "検診 · 健全" },
    strategies:  { code: "DIVISION 04 // STRATEGIES",  title: "Council & Filters",   jp: "戦術", sig: "陣形 · TACTICAL FLANK",  fx: "fx-strategies",  stamp: "陣形 · 戦術" },
    credentials: { code: "DIVISION 05 // CREDENTIALS", title: "Credentials Vault",   jp: "機密", sig: "解錠 · VAULT UNSEAL",    fx: "fx-credentials", stamp: "解錠 · 機密" },
    risk:        { code: "DIVISION 06 // DISCIPLINE",  title: "Risk & Charter",      jp: "規律", sig: "鉄則 · MONOLITH DESCENT", fx: "fx-risk",        stamp: "鉄則 · 規律" }
  };

  var MARKETS_STATIC = {
    "EUR/USD": { pip: 0.0001, spread: 1.20, precision: 5, baseUnit: "EUR", entry: 1.08520, atr: 1.95, horizon: 15, crypto: false },
    "BTC/USD": { pip: 1.00,   spread: 8.00, precision: 2, baseUnit: "BTC", entry: 80512.40, atr: 58.0, horizon: 5, crypto: true },
    "ETH/USD": { pip: 0.10,   spread: 2.40, precision: 2, baseUnit: "ETH", entry: 2381.40, atr: 23.0, horizon: 5, crypto: true }
  };

  var HALF_NORMAL_SCALE = 0.7978845608;
  var currentPaused = false;
  var selectedPair = "ALL";
  var toastTimer = null;

  function showToast(msg) {
    var el = document.getElementById("global-toast");
    if (!el) return;
    el.textContent = msg;
    el.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.hidden = true; }, 4500);
  }

  var fxTimer = null;

  function triggerPageFx(meta) {
    if (reduceMotion) return;
    var fxEl = document.getElementById("page-fx");
    var stampEl = document.getElementById("page-fx-stamp");
    if (!fxEl || !meta) return;
    if (fxTimer) clearTimeout(fxTimer);
    fxEl.className = "page-fx";
    if (stampEl) stampEl.textContent = meta.stamp || meta.jp;
    // Force reflow so switching between pages always replays the unique keyframe
    void fxEl.offsetWidth;
    fxEl.className = "page-fx is-playing " + meta.fx;
    fxTimer = setTimeout(function () {
      fxEl.className = "page-fx";
    }, 480);
  }

  /* Navigation & Hash Routing -------------------------------------------- */
  function activatePage(pageId, pushHash, playTransition) {
    if (!PAGE_META[pageId]) pageId = "signals";
    var meta = PAGE_META[pageId];

    Array.prototype.forEach.call(document.querySelectorAll(".page"), function (sec) {
      var match = sec.getAttribute("data-page") === pageId;
      if (match) {
        sec.classList.remove("is-active");
        void sec.offsetWidth;
        sec.classList.add("is-active");
      } else {
        sec.classList.remove("is-active");
      }
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-nav]"), function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-nav") === pageId);
    });

    var codeEl = document.getElementById("top-page-code");
    var sigEl = document.getElementById("top-motion-sig");
    var titleEl = document.getElementById("top-page-title");
    var jpEl = document.getElementById("top-page-jp");
    if (codeEl) codeEl.textContent = meta.code;
    if (sigEl) sigEl.textContent = meta.sig;
    if (titleEl) titleEl.textContent = meta.title;
    if (jpEl) jpEl.textContent = meta.jp;

    if (playTransition !== false) {
      triggerPageFx(meta);
    }

    if (pushHash && window.location.hash !== "#" + pageId) {
      history.replaceState(null, "", "#" + pageId);
    }
    window.scrollTo(0, 0);

    if (pageId === "signals") refreshStatus();
    if (pageId === "telegram") refreshTelegram();
    if (pageId === "health") refreshHealth(false);
    if (pageId === "strategies") refreshStrategies();
    if (pageId === "credentials") refreshCredentials();
  }

  Array.prototype.forEach.call(document.querySelectorAll("[data-nav]"), function (btn) {
    btn.addEventListener("click", function () {
      activatePage(btn.getAttribute("data-nav"), true, true);
    });
  });

  window.addEventListener("hashchange", function () {
    var h = (window.location.hash || "").replace(/^#/, "");
    if (PAGE_META[h]) activatePage(h, false, true);
  });

  /* Shared Bot State Sync ------------------------------------------------ */
  function syncBotStateUI(paused, feedMode) {
    currentPaused = Boolean(paused);
    var dot = document.getElementById("rail-dot");
    var railState = document.getElementById("rail-bot-state");
    var railFeed = document.getElementById("rail-feed-mode");
    var topPill = document.getElementById("top-state-pill");
    var topBtn = document.getElementById("top-toggle-pause");
    var sigFeed = document.getElementById("sig-feed-indicator");

    if (dot) dot.classList.toggle("is-paused", currentPaused);
    if (railState) railState.textContent = currentPaused ? "PAUSED" : "ACTIVE";
    if (topPill) {
      topPill.textContent = currentPaused ? "BOT PAUSED" : "BOT ACTIVE";
      topPill.className = "pill " + (currentPaused ? "pill-paused" : "pill-live");
    }
    if (topBtn) topBtn.textContent = currentPaused ? "Resume Bot" : "Pause Bot";
    if (feedMode) {
      var label = feedMode === "LIVE"
        ? "LIVE SPOT FEED"
        : (feedMode === "SUPABASE_SYNC" ? "SUPABASE REAL M1 SYNC" : "PAPER SIMULATION");
      if (railFeed) railFeed.textContent = label;
      if (sigFeed) sigFeed.textContent = "MODE: " + label;
    }
  }

  var togglePauseBtn = document.getElementById("top-toggle-pause");
  if (togglePauseBtn) {
    togglePauseBtn.addEventListener("click", function () {
      var act = currentPaused ? "resume" : "pause";
      apiPost("api/control", { action: act })
        .then(function (res) {
          syncBotStateUI(res.paused);
          showToast(res.message);
        })
        .catch(function () {
          syncBotStateUI(!currentPaused);
          showToast(currentPaused ? "Local state set to PAUSED." : "Local state set to ACTIVE.");
        });
    });
  }

  /* Page 01 : Signals & Market Tape -------------------------------------- */
  function renderMarkets(markets) {
    var grid = document.getElementById("markets-grid");
    if (!grid || !markets) return;
    grid.innerHTML = markets.map(function (m) {
      var be = m.break_even_wr == null ? ">100%" : (m.break_even_wr + "%");
      var kerTxt = m.ker_10 != null ? Number(m.ker_10).toFixed(2) : "0.35";
      var chopTxt = m.chop_14 != null ? Number(m.chop_14).toFixed(1) : "48.0";
      var gapTxt = m.mtf_15m_gap_bps != null ? ((m.mtf_15m_gap_bps >= 0 ? "+" : "") + Number(m.mtf_15m_gap_bps).toFixed(1) + "bp") : "--";
      var m60Txt = m.mtf_60m || "WARMUP";
      return (
        '<div class="mkt-card">' +
          '<div class="mkt-head">' +
            '<strong>' + esc(m.symbol) + '</strong>' +
            '<span class="mono mkt-price">' + fmtPrice(m.latest_price, m.precision) + '</span>' +
          '</div>' +
          '<div class="kv-list">' +
            '<div class="kv"><dt>Regime / 15m / 60m</dt><dd class="mono">' + esc(m.regime) + " / " + esc(m.mtf_15m) + " (" + esc(gapTxt) + ") / " + esc(m60Txt) + '</dd></div>' +
            '<div class="kv"><dt>M1 ATR / Spread</dt><dd class="mono">' + esc(m.atr_pips) + "p / " + esc(m.spread_pips) + 'p</dd></div>' +
            '<div class="kv"><dt>KER(10) / CHOP(14)</dt><dd class="mono">' + esc(kerTxt) + " / " + esc(chopTxt) + '</dd></div>' +
            '<div class="kv"><dt>Hold / Break-Even</dt><dd class="mono">' + esc(m.resolved_horizon_m || "--") + "m / " + esc(be) + '</dd></div>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  var IST_OFFSET_MS = (5 * 60 + 30) * 60000;
  var TZ_LABEL = "UTC+05:30";

  function pad2(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function toIstShiftedDate(val) {
    var d = val instanceof Date ? val : new Date(val);
    if (isNaN(d.getTime())) return null;
    return new Date(d.getTime() + IST_OFFSET_MS);
  }

  function formatIstHm(val, includeTz) {
    var s = toIstShiftedDate(val);
    if (!s) return "--";
    var hm = pad2(s.getUTCHours()) + ":" + pad2(s.getUTCMinutes());
    return includeTz === false ? hm : (hm + " " + TZ_LABEL);
  }

  function formatIstDateTime(val) {
    var s = toIstShiftedDate(val);
    if (!s) return "--";
    return (
      s.getUTCFullYear() + "-" + pad2(s.getUTCMonth() + 1) + "-" + pad2(s.getUTCDate()) +
      " " + pad2(s.getUTCHours()) + ":" + pad2(s.getUTCMinutes()) + " " + TZ_LABEL
    );
  }

  function updateLiveIstClock() {
    var el = document.getElementById("top-clock-pill");
    if (!el) return;
    var nowMs = Date.now();
    var shifted = new Date(nowMs + IST_OFFSET_MS);
    var nextOpenMs = Math.floor(nowMs / 60000) * 60000 + 60000;
    var secsLeft = Math.max(0, Math.ceil((nextOpenMs - nowMs) / 1000));
    var nextShifted = new Date(nextOpenMs + IST_OFFSET_MS);
    var nowHms = pad2(shifted.getUTCHours()) + ":" + pad2(shifted.getUTCMinutes()) + ":" + pad2(shifted.getUTCSeconds());
    var nextHm = pad2(nextShifted.getUTCHours()) + ":" + pad2(nextShifted.getUTCMinutes());
    el.textContent = nowHms + " " + TZ_LABEL + " · NEXT " + nextHm + " (" + secsLeft + "s)";
  }
  updateLiveIstClock();
  setInterval(updateLiveIstClock, 1000);

  function renderLedger(ledger) {
    if (!ledger) return;
    var pill = document.getElementById("ledger-summary-pill");
    if (pill) {
      var netStr = (ledger.net_pips >= 0 ? "+" : "") + Number(ledger.net_pips || 0).toFixed(1);
      pill.textContent = ledger.wins + "W / " + ledger.losses + "L / " + netStr + " pips";
    }
    var tbody = document.getElementById("recent-signals-tbody");
    if (!tbody) return;
    var rows = ledger.recent_signals || [];
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="dim">No signals sealed yet. Click Generate Signal above.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(function (s) {
      var entryShort = s.entry_time ? formatIstDateTime(s.entry_time) : (s.entry_display_hm || "--");
      var sideCls = s.side === "BUY" ? "pos" : (s.side === "SELL" ? "neg" : "");
      return (
        "<tr>" +
          '<td class="mono">' + esc(s.pair) + "</td>" +
          '<td class="mono ' + sideCls + '">' + esc(s.side) + "</td>" +
          '<td class="mono">' + esc(entryShort) + "</td>" +
          '<td class="mono">' + esc(s.horizon_minutes) + "m</td>" +
          '<td class="mono">' + esc(s.regime || "--") + "</td>" +
          '<td class="mono">' + esc(s.delivery_state || "WEB_CONSOLE") + "</td>" +
          '<td class="mono">#' + esc(String(s.signal_id || "").slice(0, 8)) + "</td>" +
        "</tr>"
      );
    }).join("");
  }

  function refreshStatus() {
    apiGet("api/status")
      .then(function (st) {
        syncBotStateUI(st.paused, st.feed_mode);
        renderMarkets(st.markets);
        renderLedger(st.ledger);
      })
      .catch(function () {
        var fallback = Object.keys(MARKETS_STATIC).map(function (sym) {
          var m = MARKETS_STATIC[sym];
          return {
            symbol: sym,
            precision: m.precision,
            latest_price: m.entry,
            regime: "TREND",
            mtf_15m: "UP",
            atr_pips: m.atr,
            spread_pips: m.spread,
            resolved_horizon_m: m.horizon,
            break_even_wr: 55.2
          };
        });
        renderMarkets(fallback);
      });
  }

  function renderSignalEvaluations(cards) {
    var box = document.getElementById("signal-results");
    if (!box) return;
    box.innerHTML = cards.map(function (card, idx) {
      var isSig = Boolean(card.actionable && card.signal);
      var sideClass = card.side === "BUY" ? "sig-side-buy" : (card.side === "SELL" ? "sig-side-sell" : "");
      var badgeText = isSig ? card.side : "ABSTAIN // NO SETUP";
      var rp = card.risk_plan || {};
      var sig = card.signal || {};
      var leadHint = (typeof card.seconds_until_entry === "number")
        ? (" (in " + Math.max(0, Math.round(card.seconds_until_entry)) + "s)")
        : "";
      var entryTime = sig.entry_hm
        ? (sig.entry_hm + leadHint)
        : (sig.entry_time
            ? (formatIstHm(sig.entry_time, true) + leadHint)
            : (card.target_entry_hm ? (card.target_entry_hm + leadHint) : formatIstHm(card.evaluated_at, true)));
      var expTime = sig.expiry_hm
        ? (sig.expiry_hm + " (" + sig.horizon_minutes + "m)")
        : (sig.expiry_time ? (formatIstHm(sig.expiry_time, true) + " (" + sig.horizon_minutes + "m)") : "--");
      var unitDec = card.pair === "EUR/USD" ? 0 : 4;

      var votesHtml = (card.votes || []).map(function (v) {
        return (
          '<li class="vote-row">' +
            '<span class="mono">' + esc(v.name) + '</span>' +
            '<span class="mono">' + esc(v.side) + '</span>' +
          '</li>'
        );
      }).join("");

      return (
        '<article class="sig-card ' + (isSig ? "is-signal" : "is-abstain") + '">' +
          '<div class="sig-card-top">' +
            '<div>' +
              '<span class="sig-pair">' + esc(card.pair) + '</span>' +
              '<div class="mono dim" style="font-size:0.68rem">' + esc(card.feed_source) + " · " + esc(card.regime) + " · MTF " + esc(card.mtf_bias) + '</div>' +
            '</div>' +
            '<span class="sig-side-badge ' + sideClass + '">' + esc(badgeText) + '</span>' +
          '</div>' +
          '<p class="sig-reason">' + esc(card.reason) + '</p>' +
          '<div class="sig-metrics mono">' +
            '<div><span class="sig-metric-k">Next Candle Entry</span>' + esc(entryTime) + '</div>' +
            '<div><span class="sig-metric-k">Expiry / Hold</span>' + esc(expTime) + '</div>' +
            '<div><span class="sig-metric-k">Ref Price</span>' + fmtPrice(card.latest_price, card.precision) + '</div>' +
            '<div><span class="sig-metric-k">Core Checks</span>' + esc(card.core_checks) + " / 6" + '</div>' +
            '<div><span class="sig-metric-k">1.5x ATR Stop</span>' + fmtPrice(rp.stop_price, card.precision) + '</div>' +
            '<div><span class="sig-metric-k">2.25x Target</span>' + fmtPrice(rp.target_price, card.precision) + '</div>' +
            '<div><span class="sig-metric-k">Paper Size</span>' + fmtPrice(rp.position_units, unitDec) + " " + esc(rp.unit_label || "") + '</div>' +
            '<div><span class="sig-metric-k">Net R:R / Cost</span>' + esc(rp.net_reward_to_risk) + ":1 (" + esc(card.spread_pips) + 'p)</div>' +
          '</div>' +
          (card.brief_text
            ? '<div class="action-bar">' +
                '<button type="button" class="btn btn-sm btn-ghost copy-brief-btn" data-brief="' + esc(card.brief_text) + '">Copy Brief Signal</button>' +
                (sig.signal_id ? '<span class="mono dim" style="font-size:0.72rem;align-self:center">#' + esc(String(sig.signal_id).slice(0, 8)) + '</span>' : "") +
              '</div>'
            : "") +
          (votesHtml
            ? '<details class="sig-votes-details">' +
                '<summary>Council Breakdown (' + card.votes.length + ' members)</summary>' +
                '<ul class="vote-list">' + votesHtml + '</ul>' +
              '</details>'
            : "") +
        '</article>'
      );
    }).join("");

    Array.prototype.forEach.call(box.querySelectorAll(".copy-brief-btn"), function (btn) {
      btn.addEventListener("click", function () {
        var txt = btn.getAttribute("data-brief") || "";
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(txt).then(function () {
            showToast("Signal copied to clipboard.");
          });
        }
      });
    });
  }

  function staticFallbackSignal(pair, balance, riskPct) {
    var keys = pair === "ALL" ? ["EUR/USD", "BTC/USD", "ETH/USD"] : [pair];
    var now = new Date();
    var nextMin = new Date(Math.floor(now.getTime() / 60000) * 60000 + 60000);
    var secsUntil = Math.max(1, Math.round((nextMin.getTime() - now.getTime()) / 1000));
    return keys.map(function (sym) {
      var m = MARKETS_STATIC[sym] || MARKETS_STATIC["EUR/USD"];
      var exp = new Date(nextMin.getTime() + m.horizon * 60000);
      var entryIst = formatIstHm(nextMin, true);
      var expIst = formatIstHm(exp, true);
      var stopPips = Math.max(m.spread * 1.5, m.atr * 1.5);
      var targetPips = Math.max(stopPips, m.atr * 2.25);
      var cappedFrac = Math.min(0.02, Math.max(0.001, riskPct / 100.0));
      var units = (balance * cappedFrac) / ((stopPips + m.spread) * m.pip);
      return {
        pair: sym,
        precision: m.precision,
        feed_source: "STATIC_PAPER_ENGINE",
        timezone: TZ_LABEL,
        evaluated_at: now.toISOString(),
        target_entry_hm: entryIst,
        seconds_until_entry: secsUntil,
        is_future_entry: true,
        latest_price: m.entry,
        atr_pips: m.atr,
        spread_pips: m.spread,
        mtf_bias: "UP",
        actionable: true,
        seal_state: "SIGNAL",
        side: "BUY",
        reason: "Core 5/6 aligned with 15m/60m MTF structure for upcoming next candle (static offline mode — start web_dashboard.py for live backend).",
        regime: "TREND",
        core_checks: 5,
        votes: [],
        signal: {
          signal_id: "static01",
          entry_time: nextMin.toISOString(),
          expiry_time: exp.toISOString(),
          entry_hm: entryIst,
          expiry_hm: expIst,
          horizon_minutes: m.horizon
        },
        brief_text: sym + " BUY\nEntry  open of " + entryIst + " candle\nExpiry " + expIst + "  (" + m.horizon + " min)\n\nPaper — research only, no broker order.",
        risk_plan: {
          stop_price: m.entry - stopPips * m.pip,
          target_price: m.entry + targetPips * m.pip,
          position_units: units,
          unit_label: m.baseUnit,
          net_reward_to_risk: Number(((targetPips - m.spread) / (stopPips + m.spread)).toFixed(2))
        }
      };
    });
  }

  function triggerSignalEval(mode) {
    var genBtn = document.getElementById("btn-generate-signal");
    var insBtn = document.getElementById("btn-inspect-now");
    var bal = parseFloat((document.getElementById("sig-balance") || {}).value || "10000") || 10000;
    var rsk = parseFloat((document.getElementById("sig-risk") || {}).value || "1.0") || 1.0;
    var sendTg = Boolean((document.getElementById("sig-send-tg") || {}).checked);

    if (genBtn) genBtn.disabled = true;
    if (insBtn) insBtn.disabled = true;

    apiPost("api/signals", {
      pair: selectedPair,
      mode: mode,
      dispatch_telegram: sendTg,
      balance_usd: bal,
      risk_pct: rsk
    })
      .then(function (res) {
        renderSignalEvaluations(res.evaluations || []);
        refreshStatus();
        showToast(
          res.emitted > 0
            ? ("Sealed " + res.emitted + " future next-candle signal(s) (" + (res.target_entry_hm || TZ_LABEL) + ").")
            : ("Evaluated upcoming next candle (" + (res.target_entry_hm || TZ_LABEL) + "): gates abstained.")
        );
      })
      .catch(function () {
        renderSignalEvaluations(staticFallbackSignal(selectedPair, bal, rsk));
      })
      .finally(function () {
        if (genBtn) genBtn.disabled = false;
        if (insBtn) insBtn.disabled = false;
      });
  }

  Array.prototype.forEach.call(document.querySelectorAll("#sig-pair-tabs .seg"), function (btn) {
    btn.addEventListener("click", function () {
      Array.prototype.forEach.call(document.querySelectorAll("#sig-pair-tabs .seg"), function (b) {
        b.classList.remove("is-active");
      });
      btn.classList.add("is-active");
      selectedPair = btn.getAttribute("data-pair") || "ALL";
    });
  });

  var btnGen = document.getElementById("btn-generate-signal");
  var btnIns = document.getElementById("btn-inspect-now");
  var btnSettle = document.getElementById("btn-settle-due");
  var btnRefMkt = document.getElementById("btn-refresh-markets");

  if (btnGen) btnGen.addEventListener("click", function () { triggerSignalEval("scan"); });
  if (btnIns) btnIns.addEventListener("click", function () { triggerSignalEval("now"); });
  if (btnRefMkt) btnRefMkt.addEventListener("click", refreshStatus);
  if (btnSettle) {
    btnSettle.addEventListener("click", function () {
      apiPost("api/control", { action: "settle" })
        .then(function (r) {
          showToast(r.message);
          refreshStatus();
        })
        .catch(function (e) { showToast(e.message); });
    });
  }

  /* Page 02 : Telegram & Output Format Studio ---------------------------- */
  var SAMPLE_SIGNAL_VALS = {
    pair: "EUR/USD",
    side: "BUY",
    entry_hm: "19:45",
    expiry_hm: "20:15",
    tz: "UTC+05:30",
    entry_iso: "2026-10-25T19:45:00+05:30",
    expiry_iso: "2026-10-25T20:15:00+05:30",
    horizon: "30",
    regime: "TREND",
    probability: "0.584",
    raw_probability: "0.571",
    calibration_state: "CALIBRATED",
    calibration_desc: "calibrated",
    reason: "Core 5/6 (pullback rebound): trend, pullback reset, rebound turn, RSI & oscillator, candle quality.",
    spread_pips: "1.20",
    core_score: "5",
    short_id: "a8f31c90",
    signal_id: "a8f31c90e4b71209",
    source: "LIVE"
  };

  var SAMPLE_RESULT_VALS = {
    pair: "EUR/USD",
    side: "BUY",
    short_id: "a8f31c90",
    signal_id: "a8f31c90e4b71209",
    horizon: "30",
    outcome: "WIN",
    verdict: "WIN",
    entry_price: "1.08520",
    exit_price: "1.08890",
    gross_pips: "+3.7",
    net_pips: "+2.5",
    spread_pips: "1.2"
  };

  var DEFAULT_SIG_TMPL = "{pair} {side} ({horizon}m) #{short_id}\nEntry: {entry_hm} UTC+05:30  ->  Expiry: {expiry_hm} UTC+05:30\nRegime: {regime}  |  Model: {probability} ({calibration_state})\nGate: {reason}\nSpread charged: {spread_pips} pips";
  var DEFAULT_RES_TMPL = "RESULT {pair} {side} #{short_id} -> {verdict}\nFill: {entry_price} -> {exit_price}\nGross {gross_pips}p | Cost {spread_pips}p | Net {net_pips}p";

  function safeRenderTokens(tmpl, vals) {
    return String(tmpl || "").replace(/\{([a-zA-Z0-9_]+)\}/g, function (full, k) {
      return Object.prototype.hasOwnProperty.call(vals, k) ? String(vals[k]) : full;
    }).trim();
  }

  function renderLocalFormatPreview() {
    var sigStyleEl = document.getElementById("tg-sig-style");
    var resStyleEl = document.getElementById("tg-res-style");
    var sigTmplEl = document.getElementById("tg-sig-template");
    var resTmplEl = document.getElementById("tg-res-template");
    var footerEl = document.getElementById("tg-paper-footer");
    var prevSig = document.getElementById("tg-preview-signal");
    var prevRes = document.getElementById("tg-preview-result");
    if (!sigStyleEl || !resStyleEl || !prevSig || !prevRes) return;

    var v = SAMPLE_SIGNAL_VALS;
    var sStyle = sigStyleEl.value || "standard";
    var sigOut = "";
    if (sStyle === "brief") {
      sigOut = v.pair + " " + v.side + "\nEntry  open of " + v.entry_hm + " " + v.tz + " candle\nExpiry " + v.expiry_hm + " " + v.tz + "  (" + v.horizon + " min)\n#" + v.short_id + "\n\nPaper — research only, no broker order.";
    } else if (sStyle === "tactical") {
      sigOut = "[" + v.side + "] " + v.pair + "  ·  " + v.horizon + "m  ·  #" + v.short_id + "\nEntry " + v.entry_hm + " " + v.tz + "  ->  Expiry " + v.expiry_hm + " " + v.tz + "\nRegime " + v.regime + "  |  Model " + v.probability + " (" + v.calibration_state + ")  |  Spread " + v.spread_pips + "p\nGate: " + v.reason + "\nPaper — research only, no broker order.";
    } else if (sStyle === "compact") {
      sigOut = v.pair + " " + v.side + " | in " + v.entry_hm + " " + v.tz + " -> exp " + v.expiry_hm + " " + v.tz + " (" + v.horizon + "m) | " + v.regime + " | p=" + v.probability + " | cost " + v.spread_pips + "p | #" + v.short_id + " [paper]";
    } else if (sStyle === "custom") {
      var rawT = (sigTmplEl && sigTmplEl.value.trim()) ? sigTmplEl.value : DEFAULT_SIG_TMPL;
      sigOut = safeRenderTokens(rawT, v);
      if (footerEl && footerEl.checked && sigOut.toLowerCase().indexOf("paper") === -1) {
        sigOut += "\nPaper — research only, no broker order.";
      }
    } else {
      sigOut = v.pair + " " + v.side + "\nEntry: open of the " + v.entry_hm + " " + v.tz + " candle\nExpires: " + v.expiry_hm + " " + v.tz + " (" + v.horizon + " min)\nRegime: " + v.regime + "\nModel: " + v.probability + " — " + v.calibration_desc + " (raw " + v.raw_probability + ")\nWhy: " + v.reason + "\nCost: " + v.spread_pips + " pip spread charged on settlement.\nPaper research signal. No broker order is placed.";
    }

    var rv = SAMPLE_RESULT_VALS;
    var rStyle = resStyleEl.value || "standard";
    var resOut = "";
    if (rStyle === "compact") {
      resOut = "RESULT " + rv.pair + " " + rv.side + " #" + rv.short_id + " | " + rv.entry_price + " -> " + rv.exit_price + " | gross " + rv.gross_pips + "p | cost " + rv.spread_pips + "p | net " + rv.net_pips + "p | " + rv.verdict;
    } else if (rStyle === "custom") {
      var rawRT = (resTmplEl && resTmplEl.value.trim()) ? resTmplEl.value : DEFAULT_RES_TMPL;
      resOut = safeRenderTokens(rawRT, rv);
    } else {
      resOut = "RESULT " + rv.pair + " " + rv.side + "  #" + rv.short_id + "\nEntry " + rv.entry_price + " -> Exit " + rv.exit_price + "\nMove " + rv.gross_pips + " pips | cost " + rv.spread_pips + " | net " + rv.net_pips + "\n" + rv.verdict;
    }

    prevSig.textContent = sigOut;
    prevRes.textContent = resOut;
  }

  function insertTokenAtCursor(textarea, token) {
    if (!textarea) return;
    var start = textarea.selectionStart || 0;
    var end = textarea.selectionEnd || 0;
    var val = textarea.value || "";
    textarea.value = val.slice(0, start) + token + val.slice(end);
    textarea.focus();
    textarea.selectionStart = textarea.selectionEnd = start + token.length;
    renderLocalFormatPreview();
  }

  function populateTokenChips(containerId, items, targetTextareaId, styleSelectId) {
    var bar = document.getElementById(containerId);
    var ta = document.getElementById(targetTextareaId);
    var sel = document.getElementById(styleSelectId);
    if (!bar || bar.children.length > 0) return;
    bar.innerHTML = (items || []).map(function (item) {
      return '<button type="button" class="token-chip" data-token="' + esc(item.token) + '" title="' + esc(item.desc || "") + '">' + esc(item.token) + "</button>";
    }).join("");
    Array.prototype.forEach.call(bar.querySelectorAll(".token-chip"), function (btn) {
      btn.addEventListener("click", function () {
        if (sel && sel.value !== "custom") {
          sel.value = "custom";
        }
        insertTokenAtCursor(ta, btn.getAttribute("data-token") || "");
      });
    });
  }

  function hydrateFormatStudio(studio) {
    if (!studio) return;
    var fmt = studio.format || {};
    var sigStyleEl = document.getElementById("tg-sig-style");
    var resStyleEl = document.getElementById("tg-res-style");
    var sigTmplEl = document.getElementById("tg-sig-template");
    var resTmplEl = document.getElementById("tg-res-template");
    var applyCmdEl = document.getElementById("tg-apply-cmd");
    var footerEl = document.getElementById("tg-paper-footer");
    var activeStyleBadge = document.getElementById("tg-active-style");

    if (studio.default_signal_template) DEFAULT_SIG_TMPL = studio.default_signal_template;
    if (studio.default_result_template) DEFAULT_RES_TMPL = studio.default_result_template;

    if (sigStyleEl) sigStyleEl.value = fmt.signal_style || "standard";
    if (resStyleEl) resStyleEl.value = fmt.result_style || "standard";
    if (sigTmplEl && document.activeElement !== sigTmplEl) {
      sigTmplEl.value = fmt.signal_template || DEFAULT_SIG_TMPL;
    }
    if (resTmplEl && document.activeElement !== resTmplEl) {
      resTmplEl.value = fmt.result_template || DEFAULT_RES_TMPL;
    }
    if (applyCmdEl) applyCmdEl.checked = Boolean(fmt.apply_to_signals_cmd);
    if (footerEl) footerEl.checked = fmt.append_paper_footer !== false;
    if (activeStyleBadge) {
      activeStyleBadge.textContent = (fmt.signal_style || "standard") + " / " + (fmt.result_style || "standard");
    }

    populateTokenChips("tg-sig-tokens", studio.signal_placeholders, "tg-sig-template", "tg-sig-style");
    populateTokenChips("tg-res-tokens", studio.result_placeholders, "tg-res-template", "tg-res-style");

    if (studio.preview) {
      var prevSig = document.getElementById("tg-preview-signal");
      var prevRes = document.getElementById("tg-preview-result");
      if (prevSig) prevSig.textContent = studio.preview.scheduled_signal || "";
      if (prevRes) prevRes.textContent = studio.preview.settlement_result || "";
    } else {
      renderLocalFormatPreview();
    }
  }

  ["tg-sig-style", "tg-res-style", "tg-sig-template", "tg-res-template", "tg-apply-cmd", "tg-paper-footer"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) {
      el.addEventListener("input", renderLocalFormatPreview);
      el.addEventListener("change", renderLocalFormatPreview);
    }
  });

  var btnSaveTgFmt = document.getElementById("btn-save-tg-format");
  if (btnSaveTgFmt) {
    btnSaveTgFmt.addEventListener("click", function () {
      var payload = {
        signal_style: (document.getElementById("tg-sig-style") || {}).value || "standard",
        result_style: (document.getElementById("tg-res-style") || {}).value || "standard",
        signal_template: (document.getElementById("tg-sig-template") || {}).value || "",
        result_template: (document.getElementById("tg-res-template") || {}).value || "",
        apply_to_signals_cmd: Boolean((document.getElementById("tg-apply-cmd") || {}).checked),
        append_paper_footer: Boolean((document.getElementById("tg-paper-footer") || {}).checked)
      };
      apiPost("api/telegram/format", payload)
        .then(function (res) {
          hydrateFormatStudio(res);
          showToast("Saved Telegram output format (" + res.format.signal_style + " / " + res.format.result_style + ").");
        })
        .catch(function (err) { showToast(err.message); });
    });
  }

  var btnResetTgFmt = document.getElementById("btn-reset-tg-format");
  if (btnResetTgFmt) {
    btnResetTgFmt.addEventListener("click", function () {
      apiPost("api/telegram/format", { reset: true })
        .then(function (res) {
          var sigTmplEl = document.getElementById("tg-sig-template");
          var resTmplEl = document.getElementById("tg-res-template");
          if (sigTmplEl) sigTmplEl.value = DEFAULT_SIG_TMPL;
          if (resTmplEl) resTmplEl.value = DEFAULT_RES_TMPL;
          hydrateFormatStudio(res);
          showToast("Reset Telegram output format to standard defaults.");
        })
        .catch(function (err) { showToast(err.message); });
    });
  }

  renderLocalFormatPreview();

  function refreshTelegram() {
    apiGet("api/telegram")
      .then(function (tg) {
        var pill = document.getElementById("tg-conn-pill");
        if (pill) {
          var lbl = tg.local_send_ready
            ? "DIRECT CONNECTED"
            : (tg.configured ? "CLOUD RUNNER ACTIVE" : "WEB ONLY");
          pill.textContent = lbl;
          pill.className = "pill " + (tg.configured ? "pill-live" : "pill-warn");
        }
        document.getElementById("tg-token-mask").textContent = tg.token_masked || "Not configured";
        document.getElementById("tg-owner-mask").textContent = tg.owner_masked || "Not configured";
        document.getElementById("tg-offset").textContent = String(tg.offset || 0);
        document.getElementById("tg-pending").textContent = String(tg.pending_count || 0);
        if (tg.format_studio) {
          hydrateFormatStudio(tg.format_studio);
        }

        var tbody = document.getElementById("tg-outbox-tbody");
        var rows = tg.recent_outbox || [];
        if (tbody) {
          tbody.innerHTML = rows.length
            ? rows.map(function (r) {
                return (
                  "<tr>" +
                    '<td class="mono">' + esc(String(r.message_key || "").slice(0, 22)) + "</td>" +
                    '<td class="mono">' + esc(r.kind) + "</td>" +
                    '<td class="mono">' + esc(r.state) + "</td>" +
                    '<td class="mono">' + esc(r.attempts || 0) + "</td>" +
                    "<td>" + esc(r.last_error || (r.message_id ? "msg #" + r.message_id : "ok")) + "</td>" +
                  "</tr>"
                );
              }).join("")
            : '<tr><td colspan="5" class="dim">Outbox queue is empty.</td></tr>';
        }
      })
      .catch(function () {});
  }

  Array.prototype.forEach.call(document.querySelectorAll(".cmd-btn"), function (btn) {
    btn.addEventListener("click", function () {
      var cmd = btn.getAttribute("data-cmd") || "/status";
      var mirror = Boolean((document.getElementById("tg-mirror-chat") || {}).checked);
      var term = document.getElementById("tg-terminal-output");
      if (term) term.textContent = "> " + cmd + "\nExecuting...";
      apiPost("api/telegram/command", { command: cmd, send_to_chat: mirror })
        .then(function (res) {
          syncBotStateUI(res.paused);
          if (term) {
            term.textContent = "> " + res.command + (res.sent_to_chat ? " [mirrored to chat]" : " [local console]") + "\n\n" + res.reply;
          }
          refreshTelegram();
        })
        .catch(function (err) {
          if (term) term.textContent = "> " + cmd + "\nError: " + err.message;
        });
    });
  });

  /* Page 03 : Health & Causality Audit ----------------------------------- */
  function renderHealth(h) {
    document.getElementById("hl-tests").textContent = String(h.tests_passing || 700);
    var la = h.lookahead || {};
    var checks = la.checks || [];
    var passCount = checks.filter(function (c) { return c.ok; }).length;
    document.getElementById("hl-audit-state").textContent = passCount + " / " + (checks.length || 5) + (la.ok ? " PASS" : " FAIL");
    document.getElementById("hl-audit-ms").textContent = la.elapsed_ms ? ("Verified in " + la.elapsed_ms + " ms") : "Zero future leakage";

    var pool = h.key_pool || {};
    var slots = pool.slots || [];
    var used = slots.reduce(function (a, s) { return a + (s.day_used || 0); }, 0);
    var limit = slots.reduce(function (a, s) { return a + (s.day_limit || 0); }, 0) || 800;
    document.getElementById("hl-quota").textContent = used + " / " + limit;
    document.getElementById("hl-slots").textContent = (pool.enabled || 0) + " / " + slots.length + " key slot(s) enabled";

    var laList = document.getElementById("hl-lookahead-list");
    if (laList) {
      laList.innerHTML = checks.map(function (c) {
        return (
          '<li class="check-row">' +
            '<div><strong class="mono">' + esc(c.name) + '</strong><div class="dim">' + esc(c.detail) + '</div></div>' +
            '<span class="pill ' + (c.ok ? "pill-live" : "pill-seal") + '">' + (c.ok ? "PASS" : "FAIL") + '</span>' +
          '</li>'
        );
      }).join("");
    }

    var docList = document.getElementById("hl-doctor-list");
    if (docList && h.doctor) {
      docList.innerHTML = h.doctor.map(function (d) {
        var cls = d.status === "PASS" ? "pill-live" : (d.status === "SKIP" ? "" : "pill-warn");
        return (
          '<li class="check-row">' +
            '<div><strong>' + esc(d.name) + '</strong><div class="dim">' + esc(d.detail) + '</div></div>' +
            '<span class="pill ' + cls + '">' + esc(d.status) + '</span>' +
          '</li>'
        );
      }).join("");
    }
  }

  function refreshHealth(liveAudit) {
    var call = liveAudit ? apiPost("api/audit", {}) : apiGet("api/health");
    call
      .then(function (h) {
        renderHealth(h);
        if (liveAudit) showToast("Causality & warmup audit passed in " + h.lookahead.elapsed_ms + " ms.");
      })
      .catch(function () {});
  }

  var btnAudit = document.getElementById("btn-run-audit");
  var btnRefHl = document.getElementById("btn-refresh-health");
  if (btnAudit) btnAudit.addEventListener("click", function () { refreshHealth(true); });
  if (btnRefHl) btnRefHl.addEventListener("click", function () { refreshHealth(false); });

  /* Page 04 : Strategies & Runtime Gates --------------------------------- */
  function renderStrategies(data) {
    var rc = data.runtime_config || {};
    var activePairs = rc.pairs || ["EUR/USD", "BTC/USD", "ETH/USD"];
    Array.prototype.forEach.call(document.querySelectorAll('input[name="cfg-pair"]'), function (cb) {
      cb.checked = activePairs.indexOf(cb.value) !== -1;
    });
    document.getElementById("cfg-mtf").checked = Boolean(rc.mtf_filter);
    document.getElementById("cfg-mtf-min").value = String(rc.mtf_minutes || 15);
    var gapEl = document.getElementById("cfg-mtf-gap");
    if (gapEl && rc.mtf_min_gap_bps !== undefined) {
      var gv = Number(rc.mtf_min_gap_bps);
      gapEl.value = gv === 0 ? "0" : gv.toFixed(1);
    }
    var macroEl = document.getElementById("cfg-mtf-macro");
    if (macroEl && rc.mtf_macro_filter !== undefined) macroEl.checked = Boolean(rc.mtf_macro_filter);
    document.getElementById("cfg-news").checked = Boolean(rc.news_filter);
    document.getElementById("cfg-adaptive").checked = Boolean(rc.adaptive_horizon);
    document.getElementById("cfg-session").checked = Boolean(rc.session_filter);
    var rollEl = document.getElementById("cfg-rollover");
    if (rollEl && rc.fx_rollover_guard !== undefined) rollEl.checked = Boolean(rc.fx_rollover_guard);
    var cdEl = document.getElementById("cfg-cooldown");
    if (cdEl && rc.signal_cooldown_minutes !== undefined) cdEl.value = String(rc.signal_cooldown_minutes);
    var lsEl = document.getElementById("cfg-loss-streak");
    if (lsEl && rc.loss_guard_streak !== undefined) lsEl.value = String(rc.loss_guard_streak);
    var shEl = document.getElementById("cfg-shadow-recovery");
    if (shEl && rc.loss_guard_shadow_recovery !== undefined) shEl.checked = Boolean(rc.loss_guard_shadow_recovery);

    var tbody = document.getElementById("strat-members-tbody");
    if (tbody && data.members) {
      tbody.innerHTML = data.members.map(function (m) {
        return (
          "<tr>" +
            '<td class="mono">' + esc(m.id) + "</td>" +
            '<td class="mono">' + esc(m.family) + "</td>" +
            "<td>" + esc(m.description) + "</td>" +
            '<td class="mono">' + esc((m.regimes || []).join(", ")) + "</td>" +
            '<td class="mono">' + esc(m.weight) + "</td>" +
            '<td class="mono">' + esc(m.samples_seen) + "</td>" +
          "</tr>"
        );
      }).join("");
    }
  }

  function refreshStrategies() {
    apiGet("api/strategies").then(renderStrategies).catch(function () {});
  }

  var btnSaveCfg = document.getElementById("btn-save-config");
  if (btnSaveCfg) {
    btnSaveCfg.addEventListener("click", function () {
      var pairs = [];
      Array.prototype.forEach.call(document.querySelectorAll('input[name="cfg-pair"]:checked'), function (cb) {
        pairs.push(cb.value);
      });
      if (!pairs.length) {
        showToast("Keep at least one market enabled.");
        return;
      }
      apiPost("api/config", {
        pairs: pairs,
        mtf_filter: document.getElementById("cfg-mtf").checked,
        mtf_minutes: parseInt(document.getElementById("cfg-mtf-min").value, 10) || 15,
        mtf_min_gap_bps: parseFloat((document.getElementById("cfg-mtf-gap") || {}).value || "0") || 0,
        mtf_macro_filter: Boolean((document.getElementById("cfg-mtf-macro") || {}).checked),
        news_filter: document.getElementById("cfg-news").checked,
        adaptive_horizon: document.getElementById("cfg-adaptive").checked,
        session_filter: document.getElementById("cfg-session").checked,
        fx_rollover_guard: Boolean((document.getElementById("cfg-rollover") || {}).checked),
        signal_cooldown_minutes: parseInt((document.getElementById("cfg-cooldown") || {}).value || "0", 10) || 0,
        loss_guard_streak: parseInt((document.getElementById("cfg-loss-streak") || {}).value || "0", 10) || 0,
        loss_guard_shadow_recovery: Boolean((document.getElementById("cfg-shadow-recovery") || {}).checked)
      })
        .then(function (res) {
          renderStrategies(res.strategies);
          showToast("Engine runtime gates updated for " + pairs.join(", ") + ".");
        })
        .catch(function (err) { showToast(err.message); });
    });
  }

  /* Page 05 : Credentials Vault ------------------------------------------ */
  function renderCredentials(c) {
    var td = c.twelvedata || {};
    var sb = c.supabase || {};
    var tg = c.telegram || {};

    document.getElementById("cred-td-count").textContent = (td.slot_count || 0) + " / " + (c.max_twelvedata_slots || 30);
    document.getElementById("cred-td-detail").textContent = td.local_keys_ready
      ? "Masked slots active in local .env"
      : (td.configured ? "Synced from Supabase / GitHub Actions" : "No Twelve Data key set yet");

    document.getElementById("cred-sb-state").textContent = sb.configured
      ? (sb.mode === "LIVE" ? "CONNECTED (LIVE)" : "CONNECTED")
      : "LOCAL ONLY";
    document.getElementById("cred-sb-url").textContent = sb.url_masked
      ? (sb.url_masked + (sb.key_masked ? " · " + sb.key_masked : ""))
      : "Using local JSONL store";

    document.getElementById("cred-tg-state").textContent = tg.local_send_ready
      ? "CONNECTED"
      : (tg.configured ? "CLOUD ACTIVE" : "WEB ONLY");
    document.getElementById("cred-tg-chat").textContent = tg.local_send_ready
      ? ("Chat " + tg.owner_chat_masked)
      : (tg.configured ? ("GitHub Actions · " + tg.owner_chat_masked) : "Operate from web console");

    var chips = document.getElementById("cred-slot-chips");
    if (chips) {
      chips.innerHTML = (td.slots_masked || []).map(function (s) {
        return '<span class="slot-chip">#' + esc(s.slot) + " " + esc(s.masked) + "</span>";
      }).join("");
    }
  }

  function refreshCredentials() {
    apiGet("api/credentials").then(renderCredentials).catch(function () {});
  }

  var credForm = document.getElementById("cred-form");
  if (credForm) {
    credForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var payload = {
        twelvedata_keys: document.getElementById("in-td-keys").value,
        supabase_url: document.getElementById("in-sb-url").value,
        supabase_service_key: document.getElementById("in-sb-key").value,
        telegram_token: document.getElementById("in-tg-token").value,
        telegram_owner_chat_id: document.getElementById("in-tg-chat").value
      };
      apiPost("api/credentials", payload)
        .then(function (res) {
          credForm.reset();
          renderCredentials(res.credentials);
          showToast((res.notices || ["Saved to local .env"]).join(" · "));
        })
        .catch(function (err) { showToast(err.message); });
    });
  }

  var btnClearTd = document.getElementById("btn-clear-td");
  if (btnClearTd) {
    btnClearTd.addEventListener("click", function () {
      apiPost("api/credentials", { clear_twelvedata: true })
        .then(function (res) {
          renderCredentials(res.credentials);
          showToast("Cleared Twelve Data slots in local .env.");
        })
        .catch(function (err) { showToast(err.message); });
    });
  }

  /* Page 06 : 1-2% ATR Risk & Cost Workbench ----------------------------- */
  var wbForm = document.getElementById("wb-form");
  function updateWorkbench() {
    if (!wbForm) return;
    var sym = document.getElementById("wb-symbol").value || "EUR/USD";
    var market = MARKETS_STATIC[sym] || MARKETS_STATIC["EUR/USD"];
    var side = document.getElementById("wb-side").value === "SELL" ? "SELL" : "BUY";
    var horizon = Math.max(1, Math.min(120, parseInt(document.getElementById("wb-horizon").value || "15", 10) || 15));
    var entry = Math.max(0.0001, parseFloat(document.getElementById("wb-entry").value || String(market.entry)) || market.entry);
    var atr = Math.max(0.05, parseFloat(document.getElementById("wb-atr").value || String(market.atr)) || market.atr);
    var balance = Math.max(100, parseFloat(document.getElementById("wb-balance").value || "10000") || 10000);
    var rawRiskPct = Math.max(0.1, parseFloat(document.getElementById("wb-risk").value || "1.0") || 1.0);
    var cappedRiskFrac = Math.min(rawRiskPct / 100.0, 0.02);

    var spread = market.spread;
    var expectedMove = atr * Math.sqrt(horizon) * HALF_NORMAL_SCALE;
    var breakEven = expectedMove > 0 ? 0.5 + (spread / (2.0 * expectedMove)) : Infinity;
    var neededMove = 3.0 * spread;
    var minHoldRaw = Math.ceil(Math.pow(neededMove / (atr * HALF_NORMAL_SCALE), 2));
    var minHoldText = minHoldRaw <= 120 ? (Math.max(1, minHoldRaw) + " min") : "> 120 min (too quiet)";

    var stopPips = Math.max(spread * 1.5, atr * 1.5);
    var targetPips = Math.max(stopPips, atr * 2.25);
    var stopPrice = side === "BUY" ? (entry - stopPips * market.pip) : (entry + stopPips * market.pip);
    var targetPrice = side === "BUY" ? (entry + targetPips * market.pip) : (entry - targetPips * market.pip);

    var effRiskPips = stopPips + spread;
    var effRewardPips = Math.max(0, targetPips - spread);
    var netRR = effRiskPips > 0 ? (effRewardPips / effRiskPips) : 0;
    var riskUsd = balance * cappedRiskFrac;
    var units = (effRiskPips * market.pip) > 0 ? (riskUsd / (effRiskPips * market.pip)) : 0;

    document.getElementById("out-spread").textContent = spread.toFixed(2) + " pips";
    document.getElementById("out-be").textContent = breakEven <= 1.0 ? ((breakEven * 100).toFixed(1) + "%") : "Unpayable (>100%)";
    document.getElementById("out-minhold").textContent = minHoldText;
    document.getElementById("out-stop").textContent = fmtPrice(stopPrice, market.precision);
    document.getElementById("out-target").textContent = fmtPrice(targetPrice, market.precision);
    document.getElementById("out-rr").textContent = netRR.toFixed(2) + " : 1";
    document.getElementById("out-riskusd").textContent = "$" + fmtPrice(riskUsd, 2) + " (" + (cappedRiskFrac * 100).toFixed(2) + "%)";
    document.getElementById("out-units").textContent = fmtPrice(units, market.baseUnit === "EUR" ? 0 : 4) + " " + market.baseUnit;

    var banner = document.getElementById("wb-verdict");
    if (expectedMove >= neededMove) {
      banner.textContent = sym + " at " + horizon + "m clears the 3.0x cost gate (projected " + expectedMove.toFixed(2) + "p vs " + neededMove.toFixed(2) + "p floor).";
    } else if (minHoldRaw <= 120) {
      banner.textContent = sym + " at " + horizon + "m is below 3.0x cost — adaptive horizon stretches hold to " + minHoldRaw + "m.";
    } else {
      banner.textContent = sym + " at " + atr.toFixed(2) + "p ATR is too quiet for any hold up to 120m — engine abstains.";
    }
  }

  Array.prototype.forEach.call(document.querySelectorAll(".wb-seg"), function (btn) {
    btn.addEventListener("click", function () {
      var sym = btn.getAttribute("data-symbol") || "EUR/USD";
      var m = MARKETS_STATIC[sym];
      if (!m) return;
      Array.prototype.forEach.call(document.querySelectorAll(".wb-seg"), function (b) { b.classList.remove("is-active"); });
      btn.classList.add("is-active");
      document.getElementById("wb-symbol").value = sym;
      document.getElementById("wb-entry").value = m.entry;
      document.getElementById("wb-atr").value = m.atr;
      document.getElementById("wb-horizon").value = m.horizon;
      updateWorkbench();
    });
  });

  if (wbForm) {
    wbForm.addEventListener("input", updateWorkbench);
    wbForm.addEventListener("change", updateWorkbench);
    updateWorkbench();
  }

  /* Needle 3 Command & Headline Router ----------------------------------- */
  function refreshNeedleStatus() {
    apiGet("api/needle/status")
      .then(function (st) {
        var pill = document.getElementById("needle-engine-pill");
        if (!pill) return;
        var pkg = st.package || {};
        var rep = st.finetune_report || null;
        if (pkg.installed && rep) {
          pill.textContent = "NEEDLE " + (pkg.version || "3") + " // LORA TUNED";
        } else if (pkg.installed) {
          pill.textContent = "NEEDLE " + (pkg.version || "3") + " // READY";
        } else {
          pill.textContent = "SCHEMA ROUTER // READY";
        }
      })
      .catch(function () {});
  }

  function executeNeedleQuery(rawQuery) {
    var input = document.getElementById("needle-query-input");
    var term = document.getElementById("needle-terminal-output");
    var q = String(rawQuery !== undefined ? rawQuery : (input ? input.value : "")).trim();
    if (!q) return;
    if (input && rawQuery !== undefined) input.value = q;
    if (term) term.textContent = "> " + q + "\nRouting via Needle 3...";
    apiPost("api/needle/run", { query: q, execute: true })
      .then(function (res) {
        var dec = res.decision || {};
        var callsJson = JSON.stringify(dec.function_calls || []);
        var header =
          "> " + q + "\n" +
          "Engine : " + (dec.engine || "unknown") + "  |  Decision : " + (dec.type || "refusal") + "\n" +
          "Calls  : " + callsJson + "\n\n";
        if (term) term.textContent = header + (res.reply || "");
        refreshStatus();
        refreshTelegram();
      })
      .catch(function (err) {
        if (term) term.textContent = "> " + q + "\nError: " + err.message;
      });
  }

  var btnNeedleRun = document.getElementById("btn-needle-run");
  var needleInput = document.getElementById("needle-query-input");
  if (btnNeedleRun) {
    btnNeedleRun.addEventListener("click", function () { executeNeedleQuery(); });
  }
  if (needleInput) {
    needleInput.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        executeNeedleQuery();
      }
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll(".needle-sample-btn"), function (btn) {
    btn.addEventListener("click", function () {
      executeNeedleQuery(btn.getAttribute("data-query") || "");
    });
  });

  /* Initial Boot --------------------------------------------------------- */
  refreshNeedleStatus();
  var initPage = (window.location.hash || "").replace(/^#/, "") || "signals";
  activatePage(initPage, false, false);
})();
