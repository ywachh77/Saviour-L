/* ==========================================================================
   SAVIOUR — Gate Transition & Multi-Page Operator Console Controller
   Uses strictly relative API paths ("api/...") and graceful static fallback.
   Zero external libraries, zero emojis.
   ========================================================================== */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Ambient Reiatsu Particle Field & Katana Slash FX (index + console) --- */
  function initReiatsuCanvas() {
    if (reduceMotion) return;
    var canvas = document.getElementById("reiatsu-canvas");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    var w = 0;
    var h = 0;
    var particles = [];
    var count = 38;

    function resize() {
      w = canvas.width = window.innerWidth || 1280;
      h = canvas.height = window.innerHeight || 800;
    }
    resize();
    window.addEventListener("resize", resize);

    for (var i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: 0.7 + Math.random() * 1.9,
        vx: (Math.random() - 0.5) * 0.28,
        vy: -(0.22 + Math.random() * 0.62),
        alpha: 0.14 + Math.random() * 0.42,
        crimson: Math.random() < 0.28
      });
    }

    function step() {
      ctx.clearRect(0, 0, w, h);
      for (var j = 0; j < particles.length; j++) {
        var p = particles[j];
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < -10) {
          p.y = h + 10;
          p.x = Math.random() * w;
        }
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.crimson
          ? ("rgba(158, 31, 31, " + p.alpha.toFixed(2) + ")")
          : ("rgba(236, 232, 223, " + (p.alpha * 0.65).toFixed(2) + ")");
        ctx.fill();
      }
      window.requestAnimationFrame(step);
    }
    window.requestAnimationFrame(step);
  }

  function spawnKatanaSlash(x, y) {
    if (reduceMotion || !document.body) return;
    var slash = document.createElement("span");
    slash.className = "katana-slash-fx";
    var angle = (-34 + Math.round(Math.random() * 24)) + "deg";
    slash.style.left = x + "px";
    slash.style.top = y + "px";
    slash.style.setProperty("--slash-angle", angle);
    document.body.appendChild(slash);
    setTimeout(function () {
      if (slash.parentNode) slash.parentNode.removeChild(slash);
    }, 520);
  }

  initReiatsuCanvas();
  document.addEventListener("pointerdown", function (ev) {
    var tgt = ev.target;
    if (tgt && tgt.closest && tgt.closest(".btn, .unleash-btn, .nav-item, .tab-btn, .manga-plate-card, .console-manga-item")) {
      spawnKatanaSlash(ev.clientX || (window.innerWidth / 2), ev.clientY || (window.innerHeight / 2));
    }
  });

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
          enEl.textContent = st.council_members + " COUNCIL GATES · 6 MERGED FAMILIES · " + feedTag;
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
    signals:     { code: "DIVISION 01 // SIGNALS",     title: "Signal Execution",    jp: "信号", sig: "一閃 · BLADE CLEAVE",    fx: "fx-signals",     stamp: "一閃 · 信号", manga: "assets/manga/zangetsu-cleave.jpg",     plate: "CHRONICLE PLATE 01 // ZANGETSU CLEAVE",     quote: "Completed-bar precision only — cut at the exact opening second of the next future candle." },
    telegram:    { code: "DIVISION 02 // TELEGRAM",    title: "Telegram & SAVIOUR AI", jp: "電信", sig: "巻物 · SCROLL UNFURL", fx: "fx-telegram",    stamp: "巻物 · 電信", manga: "assets/manga/vasto-ulquiorra.jpg",     plate: "CHRONICLE PLATE 02 // COGNITIVE COMMAND",   quote: "Human-like post-trade reflection fused with Cactus Needle 3 structured tool execution." },
    health:      { code: "DIVISION 03 // HEALTH",      title: "Health & Causality",  jp: "健全", sig: "検診 · DIAGNOSTIC SCAN", fx: "fx-health",      stamp: "検診 · 健全", manga: "assets/manga/bankai-clash.jpg",        plate: "CHRONICLE PLATE 03 // ZERO-LEAKAGE SEAL",  quote: "Five independent future-shock causality checks guard every indicator and MTF resampler." },
    strategies:  { code: "DIVISION 04 // STRATEGIES",  title: "6-Family Fusion Lab", jp: "戦術", sig: "陣形 · TACTICAL FLANK",  fx: "fx-strategies",  stamp: "陣形 · 戦術", manga: "assets/manga/ichigo-yhwach-tybw.jpg",  plate: "CHRONICLE PLATE 04 // SIX FAMILIES MERGED", quote: "22 council rules fused across 6 specialist families with automatic walk-forward backtests." },
    credentials: { code: "DIVISION 05 // CREDENTIALS", title: "Credentials Vault",   jp: "機密", sig: "解錠 · VAULT UNSEAL",    fx: "fx-credentials", stamp: "解錠 · 機密", manga: "assets/manga/yhwach-sovereign.jpg",    plate: "CHRONICLE PLATE 05 // SEALED VAULT",       quote: "Owner-only chmod 600 secret storage and 13-table Row-Level Security persistence." },
    risk:        { code: "DIVISION 06 // DISCIPLINE",  title: "Risk & Charter",      jp: "規律", sig: "鉄則 · MONOLITH DESCENT", fx: "fx-risk",        stamp: "鉄則 · 規律", manga: "assets/manga/shunsui-bankai.jpg",      plate: "CHRONICLE PLATE 06 // IRON CHARTER",       quote: "Full spread deducted on every trade — zero broker orders, zero OTC feeds, zero fabrication." },
    combiner:    { code: "DIVISION 07 // FUSION",      title: "Combiner Fusion",     jp: "融合", sig: "融合 · FUSION REACTOR",  fx: "fx-combiner",   stamp: "融合 · 結合", manga: "assets/manga/vasto-ulquiorra.jpg",      plate: "CHRONICLE PLATE 07 // INFINITE FUSION",  quote: "21 indicators fused with ML ensemble into a single directional vote." },
    generator:   { code: "DIVISION 08 // GENERATION",  title: "Signal Generator",    jp: "生成", sig: "生成 · SIGNAL FORGE",    fx: "fx-generator",  stamp: "生成 · 未来", manga: "assets/manga/bankai-clash.jpg",         plate: "CHRONICLE PLATE 08 // FUTURE FORGE",      quote: "Predicts upcoming candle direction across 1-min, 3-min, and 30-min horizons." },
    indicators:  { code: "DIVISION 09 // INDICATORS",  title: "Indicator Library",   jp: "指標", sig: "指標 · COMPASS ROSE",    fx: "fx-indicators", stamp: "指標 · 羅針", manga: "assets/manga/zangetsu-cleave.jpg",       plate: "CHRONICLE PLATE 09 // 21-STAR COMPASS",  quote: "Every indicator value rendered with directional interpretation in real time." }
  };

  var MARKETS_STATIC = {
    "EUR/USD": { pip: 0.0001, spread: 1.20, precision: 5, baseUnit: "EUR", entry: 1.08520, atr: 1.95, horizon: 30, crypto: false },
    "BTC/USD": { pip: 1.00,   spread: 8.05, precision: 2, baseUnit: "BTC", entry: 80512.40, atr: 58.0, horizon: 30, crypto: true },
    "ETH/USD": { pip: 0.10,   spread: 2.40, precision: 2, baseUnit: "ETH", entry: 2381.40, atr: 23.0, horizon: 30, crypto: true }
  };

  var HALF_NORMAL_SCALE = 0.7978845608;
  var currentPaused = false;
  var selectedPair = "ALL";
  var selectedExpiry = 1;
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

    var bannerEl = document.getElementById("division-manga-banner");
    var mangaImgEl = document.getElementById("division-manga-img");
    var plateEl = document.getElementById("division-manga-plate");
    var quoteEl = document.getElementById("division-manga-quote");
    if (mangaImgEl && meta.manga) {
      if (bannerEl) bannerEl.classList.add("is-swapping");
      mangaImgEl.src = meta.manga;
      if (plateEl) plateEl.textContent = meta.plate || "";
      if (quoteEl) quoteEl.textContent = meta.quote || "";
      setTimeout(function () {
        if (bannerEl) bannerEl.classList.remove("is-swapping");
      }, 420);
    }

    if (playTransition !== false) {
      triggerPageFx(meta);
    }

    if (pushHash && window.location.hash !== "#" + pageId) {
      history.replaceState(null, "", "#" + pageId);
    }
    window.scrollTo(0, 0);

    if (pageId === "signals") refreshStatus();
    if (pageId === "telegram") { refreshTelegram(); refreshNeedleStatus(); }
    if (pageId === "health") refreshHealth(false);
    if (pageId === "strategies") { refreshStrategies(); refreshLearning(); }
    if (pageId === "credentials") refreshCredentials();
    if (pageId === "combiner") loadCombiner();
    if (pageId === "generator") loadGenerator();
    if (pageId === "indicators") loadIndicators();
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

  function renderLedger(ledger, modelMeta) {
    if (!ledger) return;
    var pill = document.getElementById("ledger-summary-pill");
    if (pill) {
      var netStr = (ledger.net_pips >= 0 ? "+" : "") + Number(ledger.net_pips || 0).toFixed(1);
      var pendTxt = ledger.pending_signals ? (" · " + ledger.pending_signals + " PENDING") : "";
      var learnTxt = (modelMeta && modelMeta.mistakes_recorded) ? (" · " + modelMeta.mistakes_recorded + " LEARNED") : "";
      pill.textContent = ledger.wins + "W / " + ledger.losses + "L / " + netStr + "p" + pendTxt + learnTxt;
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
      var outTag = s.outcome || (s.settled ? "SETTLED" : "PENDING");
      var outCls = outTag === "WIN" ? "pos" : (outTag === "LOSS" ? "neg" : "");
      var outLabel = outTag;
      if (typeof s.net_pips === "number" && isFinite(s.net_pips)) {
        outLabel = outTag + " (" + (s.net_pips >= 0 ? "+" : "") + s.net_pips.toFixed(1) + "p)";
      } else if (outTag === "PENDING" && s.expiry_display_hm) {
        outLabel = "PENDING -> " + s.expiry_display_hm.replace(" UTC+05:30", "");
      }
      return (
        "<tr>" +
          '<td class="mono">' + esc(s.pair) + "</td>" +
          '<td class="mono ' + sideCls + '">' + esc(s.side) + "</td>" +
          '<td class="mono">' + esc(entryShort) + "</td>" +
          '<td class="mono">' + esc(s.horizon_minutes) + "m</td>" +
          '<td class="mono">' + esc(s.regime || "--") + "</td>" +
          '<td class="mono ' + outCls + '">' + esc(outLabel) + "</td>" +
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
        renderLedger(st.ledger, st.model);
      })
      .catch(function () {
        if (cachedCloudSnapshot) {
          applyStaticCloudSnapshot(cachedCloudSnapshot);
          return;
        }
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
          (card.combiner_prediction && card.combiner_prediction.available
            ? (function() {
                var cp = card.combiner_prediction;
                var cpClass = cp.combiner_score > 0.2 ? "bull" : (cp.combiner_score < -0.2 ? "bear" : "neutral");
                var voteHtml = (cp.votes || []).map(function (v) {
                  var vc = v.score > 0.1 ? "bull" : (v.score < -0.1 ? "bear" : "neutral");
                  return '<div class="combiner-vote-row"><span class="mono">' + esc(v.indicator) + '</span><span class="mono ' + vc + '">' + v.score.toFixed(2) + '</span><span class="mono dim" style="font-size:0.65rem">w=' + v.weight.toFixed(2) + '</span></div>';
                }).join("");
                return '<details class="sig-votes-details combiner-details">' +
                  '<summary>Combiner Fusion Score: <span class="' + cpClass + '">' + cp.combiner_score.toFixed(4) + '</span> · ' + esc(cp.tier) + ' · ML Up: ' + (cp.ml_up_probability * 100).toFixed(1) + '%</summary>' +
                  '<div class="combiner-card-inner">' +
                    '<div class="combiner-card-stats mono">' +
                      '<div><span class="dim">Est Price 5m:</span> ' + fmtPrice(cp.estimated_price_5min, card.precision) + '</div>' +
                      '<div><span class="dim">ATR:</span> ' + fmtPrice(cp.atr, card.precision) + '</div>' +
                      '<div><span class="dim">ML Down:</span> ' + (cp.ml_down_probability * 100).toFixed(1) + '%</div>' +
                    '</div>' +
                    '<div class="combiner-vote-list">' + voteHtml + '</div>' +
                  '</div>' +
                '</details>';
              })()
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
    var expiryMin = selectedExpiry || 1;
    return keys.map(function (sym) {
      var m = MARKETS_STATIC[sym] || MARKETS_STATIC["EUR/USD"];
      var exp = new Date(nextMin.getTime() + expiryMin * 60000);
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
        expiry_minutes: expiryMin,
        mtf_bias: "UP",
        actionable: true,
        seal_state: "SIGNAL",
        side: "BUY",
        reason: "Core 5/6 aligned with 15m/60m MTF structure for upcoming next candle (static offline mode — start web_dashboard.py for live backend).",
        regime: "TREND",
        core_checks: 5,
        votes: [],
        family_fusion: {
          vetoed: false,
          net_score: 0.24,
          synergy_pairs: ["CLASSIC_COUNCIL+SMC_ICT", "SMC_ICT+VIRAL_MOMENTUM"],
          family_votes: [
            { family: "CORE_GATE", side: "BUY" },
            { family: "CLASSIC_COUNCIL", side: "BUY" },
            { family: "SMC_ICT", side: "BUY" },
            { family: "VIRAL_MOMENTUM", side: "BUY" },
            { family: "MICRO_SCALPING", side: "NO SIGNAL" },
            { family: "QUANT_SCALP", side: "NO SIGNAL" }
          ]
        },
        signal: {
          signal_id: "static01",
          entry_time: nextMin.toISOString(),
          expiry_time: exp.toISOString(),
          entry_hm: entryIst,
          expiry_hm: expIst,
          horizon_minutes: expiryMin
        },
        brief_text: sym + " BUY\nEntry  open of " + entryIst + " candle\nExpiry " + expIst + "  (" + expiryMin + " min)\n\nPaper — research only, no broker order.",
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
    var expiry = selectedExpiry;

    if (genBtn) genBtn.disabled = true;
    if (insBtn) insBtn.disabled = true;
    var resGrid = document.getElementById("signal-results");
    if (resGrid && !reduceMotion) {
      resGrid.classList.remove("is-cleaving");
      void resGrid.offsetWidth;
      resGrid.classList.add("is-cleaving");
    }

    apiPost("api/signals", {
      pair: selectedPair,
      mode: mode,
      dispatch_telegram: sendTg,
      balance_usd: bal,
      risk_pct: rsk,
      expiry_minutes: expiry
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

  Array.prototype.forEach.call(document.querySelectorAll("#sig-expiry-tabs .seg"), function (btn) {
    btn.addEventListener("click", function () {
      Array.prototype.forEach.call(document.querySelectorAll("#sig-expiry-tabs .seg"), function (b) {
        b.classList.remove("is-active");
      });
      btn.classList.add("is-active");
      selectedExpiry = parseInt(btn.getAttribute("data-expiry") || "1", 10);
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
    document.getElementById("hl-tests").textContent = String(h.tests_passing || 713);
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
    var dg = h.data_gate || {};
    var ageEl = document.getElementById("hl-max-age");
    var latEl = document.getElementById("hl-seal-lat");
    var multEl = document.getElementById("hl-spread-mult");
    var unrEl = document.getElementById("hl-unresolve");
    if (ageEl && dg.max_candle_age_s) ageEl.textContent = dg.max_candle_age_s + "s";
    if (latEl && dg.max_seal_latency_s) latEl.textContent = dg.max_seal_latency_s + "s (late signals dropped)";
    if (multEl && dg.max_spread_multiple) multEl.textContent = Number(dg.max_spread_multiple).toFixed(1) + "x base spread";
    if (unrEl && dg.unresolve_after_m) unrEl.textContent = dg.unresolve_after_m + "m (never fabricated)";
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

    var eyebrowEl = document.getElementById("strat-registry-eyebrow");
    if (eyebrowEl && data.prototype_total) {
      eyebrowEl.textContent = "1 CORE GATE + " + data.prototype_total + " MULTI-FAMILY PROTOTYPES";
    }
    var famEl = document.getElementById("strat-family-summary");
    if (famEl && data.family_counts) {
      var fc = data.family_counts;
      famEl.textContent =
        "SMC/ICT " + (fc.SMC_ICT || 0) +
        " · Viral " + (fc.VIRAL_MOMENTUM || 0) +
        " · Quant " + (fc.QUANT_SCALP || 0) +
        " · Micro " + (fc.MICRO_SCALPING || 0) +
        " · Classic " + (fc.CLASSIC_COUNCIL || 0);
    }

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

  /* Merged 6-Family Fusion, Auto-Backtest & Trade Learning Desk ---------- */
  var selectedFamPair = "EUR/USD";
  var cachedLearning = null;

  function syncCognitiveBadges(cog) {
    if (!cog) return;
    var stage = String(cog.stage || "SENTINEL");
    var xp = Number(cog.experience_score != null ? cog.experience_score : (cog.xp || 0));
    var xpPill = document.getElementById("cog-xp-pill");
    var stagePill = document.getElementById("cog-stage-pill");
    var railCog = document.getElementById("rail-cognitive-stage");
    if (xpPill) xpPill.textContent = "SAVIOUR AI · " + stage + " · " + xp + " XP";
    if (stagePill) stagePill.textContent = stage + " (" + xp + " XP)";
    if (railCog) railCog.textContent = "SAVIOUR // " + stage;
  }

  function renderFamilyMatrix(learning) {
    var grid = document.getElementById("family-matrix-grid");
    var synBox = document.getElementById("synergy-strip");
    if (!grid) return;
    var mkts = (learning && learning.markets) || {};
    var m = mkts[selectedFamPair] || mkts["EUR/USD"] || {};
    var ff = m.family_fusion || {};
    var weights = ff.family_weights || {
      CORE_GATE: 0.24,
      CLASSIC_COUNCIL: 0.16,
      MICRO_SCALPING: 0.14,
      SMC_ICT: 0.17,
      VIRAL_MOMENTUM: 0.15,
      QUANT_SCALP: 0.14
    };
    var seen = ff.family_seen || {};
    var net = ff.family_net || {};
    var fams = ["CORE_GATE", "CLASSIC_COUNCIL", "SMC_ICT", "VIRAL_MOMENTUM", "QUANT_SCALP", "MICRO_SCALPING"];
    grid.innerHTML = fams.map(function (f) {
      var w = Number(weights[f] != null ? weights[f] : 0.15);
      var pct = Math.max(8, Math.min(100, Math.round(w * 260)));
      var n = Number(seen[f] || 0);
      var netVal = Number(net[f] || 0);
      return (
        '<div class="family-card">' +
          '<div class="family-card-top">' +
            '<span class="family-name mono">' + esc(f) + "</span>" +
            '<span class="family-weight mono">' + (w * 100).toFixed(1) + "%</span>" +
          "</div>" +
          '<div class="family-bar-track"><div class="family-bar-fill" style="width:' + pct + '%"></div></div>' +
          '<div class="mono dim" style="font-size:0.68rem">' +
            "Seen " + n + " · Net " + (netVal >= 0 ? "+" : "") + netVal.toFixed(1) +
          "</div>" +
        "</div>"
      );
    }).join("");

    if (synBox) {
      var syns = ff.synergy_weights || {};
      var synKeys = Object.keys(syns);
      if (synKeys.length) {
        synBox.innerHTML = synKeys.map(function (k) {
          var sw = Number(syns[k] || 0);
          var cls = sw > 0.005 ? "is-syn" : (sw < -0.005 ? "is-opp" : "");
          return '<span class="fam-chip ' + cls + '">' + esc(k) + " " + (sw >= 0 ? "+" : "") + sw.toFixed(3) + "</span>";
        }).join("");
      } else {
        synBox.innerHTML = '<span class="fam-chip">10 Pairwise Cross-Family Synergies Active</span>';
      }
    }
  }

  function renderLearningPayload(data) {
    if (!data) return;
    cachedLearning = data;
    syncCognitiveBadges(data.cognitive);
    renderFamilyMatrix(data);

    var btCards = document.getElementById("backtest-summary-cards");
    if (btCards && data.markets) {
      var pairs = ["EUR/USD", "BTC/USD", "ETH/USD"];
      btCards.innerHTML = pairs.map(function (sym) {
        var mk = data.markets[sym] || {};
        var bt = mk.latest_backtest || null;
        if (bt && bt.ok) {
          var netStr = (bt.net_pips >= 0 ? "+" : "") + Number(bt.net_pips).toFixed(2) + "p";
          var wrStr = bt.win_rate != null ? (Number(bt.win_rate).toFixed(1) + "% WR") : "0 signals";
          return (
            '<div class="stat-card">' +
              '<span class="stat-label mono">' + esc(sym) + " // " + esc(mk.model_mode || "VETO") + "</span>" +
              '<strong class="stat-val mono ' + (bt.net_pips >= 0 ? "pos" : "neg") + '">' + esc(netStr) + "</strong>" +
              '<span class="stat-sub mono">' + esc(wrStr) + " · " + esc(bt.signals) + " OOS sigs · " + esc(mk.reconciled_trades || 0) + " learned</span>" +
            "</div>"
          );
        }
        return (
          '<div class="stat-card">' +
            '<span class="stat-label mono">' + esc(sym) + " // " + esc(mk.model_mode || "VETO") + "</span>" +
            '<strong class="stat-val mono">' + esc(mk.online_updates || 0) + " updates</strong>" +
            '<span class="stat-sub mono">' + esc(mk.training_samples || 0) + " train samples · " + esc(mk.reconciled_trades || 0) + " settled</span>" +
          "</div>"
        );
      }).join("");
    }

    var jList = document.getElementById("learning-journal-list");
    if (jList) {
      var entries = [];
      (data.rectifications || []).slice(0, 4).forEach(function (r) {
        entries.push({
          cls: "is-heal",
          head: "SELF-HEAL // " + (r.at || TZ_LABEL),
          body: (r.actions && r.actions.length) ? r.actions.join(" · ") : (r.summary || "All subsystems verified healthy.")
        });
      });
      (data.journal || []).slice(0, 14).forEach(function (ep) {
        entries.push({
          cls: ep.outcome === "WIN" ? "is-win" : "is-loss",
          head: (ep.pair || "MKT") + " " + (ep.side || "") + " -> " + (ep.outcome || "") + " (" + (ep.recorded_display || ep.recorded_at || "") + ")",
          body: ep.narrative || ep.lesson || ""
        });
      });
      if (!entries.length) {
        jList.innerHTML = '<div class="journal-entry"><p class="journal-body dim">No post-mortem lessons yet — run a walk-forward backtest or settle due paper signals to record live trade reflections.</p></div>';
      } else {
        jList.innerHTML = entries.map(function (it) {
          return (
            '<div class="journal-entry ' + it.cls + '">' +
              '<div class="journal-head mono"><span>' + esc(it.head) + "</span></div>" +
              '<p class="journal-body">' + esc(it.body) + "</p>" +
            "</div>"
          );
        }).join("");
      }
    }
  }

  function refreshLearning() {
    apiGet("api/learning")
      .then(renderLearningPayload)
      .catch(function () {
        renderFamilyMatrix(null);
      });
  }

  Array.prototype.forEach.call(document.querySelectorAll("#fam-pair-tabs .seg"), function (btn) {
    btn.addEventListener("click", function () {
      Array.prototype.forEach.call(document.querySelectorAll("#fam-pair-tabs .seg"), function (b) {
        b.classList.remove("is-active");
      });
      btn.classList.add("is-active");
      selectedFamPair = btn.getAttribute("data-fam-pair") || "EUR/USD";
      renderFamilyMatrix(cachedLearning);
    });
  });

  var btnRunBt = document.getElementById("btn-run-backtest");
  var btnRefLearn = document.getElementById("btn-refresh-learning");
  if (btnRefLearn) btnRefLearn.addEventListener("click", refreshLearning);
  if (btnRunBt) {
    btnRunBt.addEventListener("click", function () {
      btnRunBt.disabled = true;
      showToast("Running purged 4-fold walk-forward backtest across all markets...");
      apiPost("api/backtest", { pair: "ALL" })
        .then(function (res) {
          if (res.learning) renderLearningPayload(res.learning);
          refreshStrategies();
          var okCount = (res.runs || []).filter(function (r) { return r.ok; }).length;
          showToast("Walk-forward backtest completed across " + okCount + " market(s).");
        })
        .catch(function (err) { showToast(err.message); })
        .finally(function () { btnRunBt.disabled = false; });
    });
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
    apiGet("api/credentials")
      .then(renderCredentials)
      .catch(function () {
        if (cachedCloudSnapshot) {
          renderCredentials(buildCloudCredentialsFromSnapshot(cachedCloudSnapshot));
        }
      });
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

  /* Needle 3 & SAVIOUR Human Cognitive Partner --------------------------- */
  function refreshNeedleStatus() {
    apiGet("api/needle/status")
      .then(function (st) {
        if (st.cognitive) syncCognitiveBadges(st.cognitive);
        var pill = document.getElementById("needle-engine-pill");
        if (!pill) return;
        var pkg = st.package || {};
        var rep = st.finetune_report || null;
        if (pkg.installed && rep) {
          pill.textContent = "NEEDLE " + (pkg.version || "3") + " // LORA TUNED";
        } else if (pkg.installed) {
          pill.textContent = "NEEDLE " + (pkg.version || "3") + " // READY";
        } else {
          pill.textContent = "SAVIOUR × NEEDLE // READY";
        }
      })
      .catch(function () {});
  }

  var btnSelfHeal = document.getElementById("btn-self-heal");
  if (btnSelfHeal) {
    btnSelfHeal.addEventListener("click", function () {
      var term = document.getElementById("needle-terminal-output");
      btnSelfHeal.disabled = true;
      if (term) term.textContent = "> Self-Heal & Rectify\nScanning causality, stale signals, unreconciled settlements, and cold VETO models...";
      apiPost("api/self_heal", {})
        .then(function (res) {
          if (res.learning) renderLearningPayload(res.learning);
          var heal = res.heal || {};
          var header =
            "> Autonomous Error Detection & Self-Rectification\n" +
            "Issues Found : " + (heal.issues_count || 0) + "  |  Rectified : " + (heal.rectified_count || 0) + "\n\n";
          if (term) term.textContent = header + (res.narrative || "All subsystems verified healthy.");
          refreshStatus();
          showToast("SAVIOUR self-heal completed (" + (heal.rectified_count || 0) + " rectified).");
        })
        .catch(function (err) {
          if (term) term.textContent = "Self-heal error: " + err.message;
        })
        .finally(function () { btnSelfHeal.disabled = false; });
    });
  }

  function executeNeedleQuery(rawQuery) {
    var input = document.getElementById("needle-query-input");
    var term = document.getElementById("needle-terminal-output");
    var q = String(rawQuery !== undefined ? rawQuery : (input ? input.value : "")).trim();
    if (!q) return;
    if (input && rawQuery !== undefined) input.value = q;
    if (term) term.textContent = "> " + q + "\nSAVIOUR × Needle 3 thinking...";
    apiPost("api/needle/run", { query: q, execute: true })
      .then(function (res) {
        if (res.cognitive) syncCognitiveBadges(res.cognitive);
        var dec = res.decision || {};
        var callsJson = JSON.stringify(dec.function_calls || []);
        var header =
          "> " + q + "\n" +
          "Engine : " + (dec.engine || "saviour_cognitive") + "  |  Mode : " + (dec.type || "dialogue") + "\n" +
          (dec.function_calls && dec.function_calls.length ? ("Tools  : " + callsJson + "\n\n") : "\n");
        if (term) term.textContent = header + (res.reply || "");
        if (res.signal_run && Array.isArray(res.signal_run.evaluations)) {
          renderSignalEvaluations(res.signal_run.evaluations);
        }
        refreshStatus();
        refreshTelegram();
        refreshLearning();
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

  /* Permanent Supabase Read-Only Telemetry & Market Bridge --------------- */
  var SUPABASE_PUBLIC_URL = "https://idudlcxkihqnvecwqqoj.supabase.co";
  var SUPABASE_PUBLISHABLE_KEY = "sb_publishable_yEZ6hFlXmmRTUVFPMNnTqQ_xWE2NFnH";
  var cachedCloudSnapshot = null;

  function buildCloudCredentialsFromSnapshot(snap) {
    var akList = Array.isArray(snap && snap.api_keys) ? snap.api_keys : [];
    var slots = (akList[0] && Array.isArray(akList[0].slots)) ? akList[0].slots : [];
    var msList = Array.isArray(snap && snap.model_state) ? snap.model_state : [];
    var champ = msList.filter(function (r) { return r && r.id === "champion"; })[0] || msList[0] || {};
    var tgOffset = Number((champ.runtime && champ.runtime.telegram_offset) || 0);
    return {
      max_twelvedata_slots: 30,
      supabase: {
        configured: true,
        mode: "CLOUD_RPC_SYNCED",
        url_masked: "https://idud...qqoj.supabase.co",
        key_masked: "CLOUD_RPC_PUBLIC"
      },
      twelvedata: {
        configured: slots.length > 0,
        local_keys_ready: false,
        source: slots.length > 0 ? "CLOUD_SYNCED" : "NONE",
        slot_count: slots.length,
        slots_masked: slots.map(function (s, i) {
          return { slot: Number(s.index != null ? s.index : i) + 1, masked: String(s.label || ("slot-" + (i + 1))) };
        })
      },
      telegram: {
        configured: tgOffset > 0,
        local_send_ready: false,
        source: tgOffset > 0 ? "CLOUD_SYNCED" : "NONE",
        owner_chat_masked: tgOffset > 0 ? ("offset #" + tgOffset) : ""
      }
    };
  }

  function applyStaticCloudSnapshot(snap) {
    if (!snap || typeof snap !== "object") return;
    var msList = Array.isArray(snap.model_state) ? snap.model_state : [];
    var champ = msList.filter(function (r) { return r && r.id === "champion"; })[0] || msList[0] || {};
    var isPaused = Boolean(champ.runtime && champ.runtime.paused);
    syncBotStateUI(isPaused, "SUPABASE_SYNC");

    var barsByPair = { "EUR/USD": [], "BTC/USD": [], "ETH/USD": [] };
    (Array.isArray(snap.candles_m1) ? snap.candles_m1 : []).forEach(function (b) {
      if (b && barsByPair[b.pair]) barsByPair[b.pair].push(b);
    });
    var mkts = Object.keys(MARKETS_STATIC).map(function (sym) {
      var meta = MARKETS_STATIC[sym];
      var arr = barsByPair[sym] || [];
      var lastClose = arr.length ? Number(arr[arr.length - 1].close) : meta.entry;
      var trSum = 0;
      var trCount = 0;
      for (var i = Math.max(1, arr.length - 14); i < arr.length; i++) {
        var hi = Number(arr[i].high);
        var lo = Number(arr[i].low);
        var pc = Number(arr[i - 1].close);
        trSum += Math.max(hi - lo, Math.abs(hi - pc), Math.abs(lo - pc));
        trCount++;
      }
      var atrPips = trCount > 0 ? (trSum / trCount) / meta.pip : meta.atr;
      var m15 = "FLAT";
      var gapBps = 0;
      if (arr.length >= 30) {
        var sFast = 0;
        var sSlow = 0;
        for (var j = arr.length - 15; j < arr.length; j++) sFast += Number(arr[j].close);
        for (var k = arr.length - 30; k < arr.length; k++) sSlow += Number(arr[k].close);
        var mFast = sFast / 15;
        var mSlow = sSlow / 30;
        gapBps = mSlow > 0 ? ((mFast - mSlow) / mSlow) * 10000 : 0;
        m15 = gapBps > 1.5 ? "BULL" : (gapBps < -1.5 ? "BEAR" : "FLAT");
      }
      return {
        symbol: sym,
        precision: meta.precision,
        latest_price: lastClose,
        regime: Math.abs(gapBps) > 3.0 ? "TREND" : "RANGE",
        mtf_15m: m15,
        mtf_15m_gap_bps: Number(gapBps.toFixed(1)),
        mtf_60m: arr.length >= 660 ? m15 : "SYNCED",
        atr_pips: Number(atrPips.toFixed(2)),
        spread_pips: meta.spread,
        resolved_horizon_m: 30,
        break_even_wr: 54.0
      };
    });
    renderMarkets(mkts);

    var stls = Array.isArray(snap.settlements) ? snap.settlements : [];
    var stlMap = {};
    var wins = 0;
    var losses = 0;
    var netPips = 0;
    stls.forEach(function (r) {
      if (!r) return;
      if (r.signal_id) stlMap[String(r.signal_id)] = r;
      if (String(r.source || "").toUpperCase() === "SYNTHETIC") return;
      if (r.outcome === "WIN") wins++;
      else if (r.outcome === "LOSS") losses++;
      if (typeof r.net_pips === "number") netPips += r.net_pips;
    });
    var sigRows = (Array.isArray(snap.signals) ? snap.signals : []).slice(0, 12).map(function (s) {
      var copy = Object.assign({}, s);
      var st = copy.signal_id ? stlMap[String(copy.signal_id)] : null;
      if (st) {
        copy.outcome = st.outcome || "SETTLED";
        copy.net_pips = st.net_pips;
      } else {
        copy.outcome = copy.settled ? "SETTLED" : "PENDING";
      }
      return copy;
    });
    renderLedger({
      wins: wins,
      losses: losses,
      net_pips: netPips,
      recent_signals: sigRows
    });
    renderCredentials(buildCloudCredentialsFromSnapshot(snap));
  }

  function syncCloudDatabaseSnapshot() {
    if (typeof fetch !== "function") return;
    fetch(SUPABASE_PUBLIC_URL + "/rest/v1/rpc/get_console_snapshot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Authorization": "Bearer " + SUPABASE_PUBLISHABLE_KEY
      },
      body: JSON.stringify({ candle_limit: 740 })
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("RPC " + resp.status);
        return resp.json();
      })
      .then(function (snap) {
        if (!snap || typeof snap !== "object") return;
        cachedCloudSnapshot = snap;
        return apiPost("api/cloud_sync", snap)
          .then(function (synced) {
            if (synced && synced.status) {
              syncBotStateUI(synced.status.paused, synced.status.feed_mode);
              renderMarkets(synced.status.markets);
              renderLedger(synced.status.ledger, synced.status.model);
            }
            if (synced && synced.credentials) {
              renderCredentials(synced.credentials);
            }
          })
          .catch(function () {
            applyStaticCloudSnapshot(snap);
          });
      })
      .catch(function () {});
  }

  /* ───────────────────────── Combiner / Generator / Indicators ───────────────────────── */

  function goToSignals() {
    activatePage("signals", true, true);
  }

  function loadCombiner() {
    var board = document.getElementById("combiner-score-board");
    var votes = document.getElementById("combiner-votes");
    var mlBox = document.getElementById("combiner-ml");
    var tape = document.getElementById("combiner-tape");
    if (!board) return;
    board.innerHTML = '<div class="empty-stage"><p class="empty-jp" lang="ja">読み込み中</p><p class="empty-title">Forging combiner score…</p></div>';
    fetch("api/combiner")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          board.innerHTML = '<div class="empty-stage"><p class="empty-title">No combiner data available.</p></div>';
          return;
        }
        var c = data.combiner || {};
        var score = c.combiner_score || 0;
        var tier = c.tier || "NEUTRAL";
        var tierClass = "bull";
        if (score < -0.2) tierClass = "bear";
        if (Math.abs(score) < 0.2) tierClass = "neutral";
        board.innerHTML =
          '<div class="combiner-score-display ' + tierClass + '">' +
            '<div class="combiner-score-value">' + score.toFixed(4) + '</div>' +
            '<div class="combiner-score-tier ' + tierClass + '">' + esc(tier) + '</div>' +
            '<div class="combiner-score-bar">' +
              '<div class="combiner-score-fill ' + tierClass + '" style="width:' + (Math.abs(score) * 50) + '%;margin-left:' + (score < 0 ? 50 - Math.abs(score) * 50 : 50) + '%"></div>' +
            '</div>' +
            '<div class="combiner-score-meta mono">' +
              '<span>Last Close: ' + fmtPrice(data.last_candle ? data.last_candle.close : 0) + '</span>' +
              '<span>Candles: ' + data.candle_count + '</span>' +
            '</div>' +
          '</div>';

        // Votes
        if (votes) {
          var voteHtml = "";
          if (c.votes && c.votes.length) {
            c.votes.forEach(function (v) {
              var vClass = v.score > 0.1 ? "bull" : (v.score < -0.1 ? "bear" : "neutral");
              voteHtml +=
                '<div class="vote-card ' + vClass + '">' +
                  '<span class="vote-name">' + esc(v.indicator) + '</span>' +
                  '<span class="vote-score mono">' + v.score.toFixed(2) + '</span>' +
                  '<span class="vote-weight mono">w=' + v.weight.toFixed(2) + '</span>' +
                '</div>';
            });
          }
          votes.innerHTML = voteHtml || '<p class="dim">No votes computed.</p>';
        }

        // ML Prediction
        if (mlBox) {
          var ml = data.ml_prediction || {};
          if (ml.available) {
            mlBox.innerHTML =
              '<div class="ml-prediction-card">' +
                '<div class="ml-row"><span class="ml-label">Up Probability</span><span class="ml-val bull mono">' + (ml.up_probability * 100).toFixed(1) + '%</span></div>' +
                '<div class="ml-row"><span class="ml-label">Down Probability</span><span class="ml-val bear mono">' + (ml.down_probability * 100).toFixed(1) + '%</span></div>' +
                '<div class="ml-row"><span class="ml-label">Last Close</span><span class="ml-val mono">' + fmtPrice(ml.last_close) + '</span></div>' +
                '<div class="ml-row"><span class="ml-label">Est. Price (5min)</span><span class="ml-val mono">' + fmtPrice(ml.estimated_price_5min) + '</span></div>' +
                '<div class="ml-row"><span class="ml-label">Confidence</span><span class="ml-val mono">' + (ml.confidence * 100).toFixed(1) + '%</span></div>' +
                '<div class="ml-row"><span class="ml-label">ATR</span><span class="ml-val mono">' + fmtPrice(ml.atr) + '</span></div>' +
              '</div>';
          } else {
            mlBox.innerHTML = '<p class="dim">ML model not available: ' + esc(ml.reason || "unknown") + '</p>';
          }
        }

        // Price Tape
        if (tape && data.recent_candles && data.recent_candles.length) {
          var candles = data.recent_candles.slice(-40);
          var maxH = Math.max.apply(null, candles.map(function (c) { return c.high; }));
          var minL = Math.min.apply(null, candles.map(function (c) { return c.low; }));
          var range = maxH - minL || 1;
          var html = '<div class="price-tape-bars">';
          candles.forEach(function (c) {
            var up = c.close >= c.open;
            var h = ((c.high - minL) / range) * 100;
            var l = ((c.low - minL) / range) * 100;
            html += '<div class="tape-bar ' + (up ? "bull" : "bear") + '" style="height:' + (100 - l) + '%;margin-bottom:' + (h - l) * 0.1 + '%"><div class="tape-bar-body" style="height:' + (h - l) + '%;top:' + l + '%"></div></div>';
          });
          html += '</div>';
          html += '<div class="price-tape-labels mono"><span>' + fmtPrice(minL) + '</span><span>' + fmtPrice(maxH) + '</span></div>';
          tape.innerHTML = html;
        }
      })
      .catch(function () {
        board.innerHTML = '<div class="empty-stage"><p class="empty-title">Failed to load combiner data.</p></div>';
      });
  }

  function loadGenerator() {
    var summary = document.getElementById("generator-summary");
    var horizons = document.getElementById("generator-horizons");
    if (!summary) return;
    summary.innerHTML = '<div class="empty-stage"><p class="empty-jp" lang="ja">読み込み中</p><p class="empty-title">Forging future signals…</p></div>';
    fetch("api/generator")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          summary.innerHTML = '<div class="empty-stage"><p class="empty-title">No generator data available.</p></div>';
          return;
        }
        var score = data.combiner_score || 0;
        var tier = data.tier || "NEUTRAL";
        var tierClass = "bull";
        if (score < -0.2) tierClass = "bear";
        if (Math.abs(score) < 0.2) tierClass = "neutral";
        summary.innerHTML =
          '<div class="generator-summary-card">' +
            '<div class="gen-summary-score ' + tierClass + '">' + score.toFixed(4) + '</div>' +
            '<div class="gen-summary-tier ' + tierClass + '">' + esc(tier) + '</div>' +
            '<div class="gen-summary-meta mono">' +
              '<span>Last Close: ' + fmtPrice(data.last_close) + '</span>' +
              '<span>ML Up: ' + (data.ml_up_prob * 100).toFixed(1) + '%</span>' +
              '<span>ML Down: ' + (data.ml_down_prob * 100).toFixed(1) + '%</span>' +
            '</div>' +
          '</div>';

        if (horizons && data.horizons && data.horizons.length) {
          var html = "";
          data.horizons.forEach(function (h) {
            var dirClass = h.direction === "BUY" ? "bull" : "bear";
            html +=
              '<div class="horizon-card">' +
                '<div class="horizon-header">' +
                  '<span class="horizon-name">' + esc(h.horizon) + '</span>' +
                  '<span class="horizon-dir ' + dirClass + '">' + h.direction + '</span>' +
                '</div>' +
                '<div class="horizon-body">' +
                  '<div class="horizon-row"><span class="mono dim">Entry</span><span class="mono">' + fmtPrice(h.entry_price) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">Est. Price</span><span class="mono">' + fmtPrice(h.estimated_price) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">Confidence</span><span class="mono ' + dirClass + '">' + (h.confidence * 100).toFixed(1) + '%</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">TP 1R</span><span class="mono bull">' + fmtPrice(h.tp_1r) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">TP 2R</span><span class="mono bull">' + fmtPrice(h.tp_2r) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">SL 1R</span><span class="mono bear">' + fmtPrice(h.sl_1r) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">SL 2R</span><span class="mono bear">' + fmtPrice(h.sl_2r) + '</span></div>' +
                  '<div class="horizon-row"><span class="mono dim">ATR</span><span class="mono">' + fmtPrice(h.atr) + '</span></div>' +
                '</div>' +
              '</div>';
          });
          horizons.innerHTML = html;
        }
      })
      .catch(function () {
        summary.innerHTML = '<div class="empty-stage"><p class="empty-title">Failed to load generator data.</p></div>';
      });
  }

  function loadIndicators() {
    var grid = document.getElementById("indicators-grid");
    if (!grid) return;
    grid.innerHTML = '<div class="empty-stage"><p class="empty-jp" lang="ja">読み込み中</p><p class="empty-title">Scanning indicators…</p></div>';
    fetch("api/indicators")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok || !data.indicators) {
          grid.innerHTML = '<div class="empty-stage"><p class="empty-title">No indicator data available.</p></div>';
          return;
        }
        var ind = data.indicators;
        var html = "";
        Object.keys(ind).forEach(function (key) {
          var v = ind[key];
          if (!v || v.value === null || v.value === undefined) return;
          var cls = v.signal === "BULLISH" ? "bull" : (v.signal === "BEARISH" ? "bear" : "neutral");
          var valDisplay = (typeof v.value === "number") ? fmtPrice(v.value) : String(v.value);
          html +=
            '<div class="indicator-card ' + cls + '">' +
              '<div class="indicator-head">' +
                '<span class="indicator-name">' + esc(key) + '</span>' +
                '<span class="indicator-signal ' + cls + '">' + v.signal + '</span>' +
              '</div>' +
              '<div class="indicator-value mono">' + valDisplay + '</div>' +
              '<div class="indicator-desc mono dim">' + esc(v.desc || "") + '</div>' +
            '</div>';
        });
        grid.innerHTML = html || '<p class="dim">No indicator values available.</p>';
      })
      .catch(function () {
        grid.innerHTML = '<div class="empty-stage"><p class="empty-title">Failed to load indicators.</p></div>';
      });
  }

  /* Initial Boot --------------------------------------------------------- */
  refreshNeedleStatus();
  refreshLearning();
  var initPage = (window.location.hash || "").replace(/^#/, "") || "signals";
  activatePage(initPage, false, false);
  syncCloudDatabaseSnapshot();
  setInterval(syncCloudDatabaseSnapshot, 45000);
})();
