/* ==========================================================================
   SAVIOUR — Shinigami Workbench & Minimal Reveal Script
   Works standalone on static GitHub Pages and enhances via relative /api/status
   when served by scripts/web_dashboard.py.
   ========================================================================== */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1. Minimal scroll reveals -------------------------------------------- */
  var reveals = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach(function (el) { el.classList.add("in"); });
  } else {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });

    reveals.forEach(function (el) { observer.observe(el); });
  }

  /* 2. Cost, Horizon & 1-2% Risk Workbench ------------------------------- */
  var MARKETS = {
    "EUR/USD": { pip: 0.0001, spread: 1.20, precision: 5, baseUnit: "EUR", entry: 1.08520, atr: 1.95, horizon: 15 },
    "BTC/USD": { pip: 1.00,   spread: 8.00, precision: 2, baseUnit: "BTC", entry: 80512.40, atr: 58.0, horizon: 5 },
    "ETH/USD": { pip: 0.10,   spread: 2.40, precision: 2, baseUnit: "ETH", entry: 2381.40, atr: 23.0, horizon: 5 }
  };

  var HALF_NORMAL_SCALE = 0.7978845608; // sqrt(2 / pi), matches fusion/costs.py

  var form = document.getElementById("wb-form");
  var symInput = document.getElementById("wb-symbol");
  var sideInput = document.getElementById("wb-side");
  var horInput = document.getElementById("wb-horizon");
  var entryInput = document.getElementById("wb-entry");
  var atrInput = document.getElementById("wb-atr");
  var balInput = document.getElementById("wb-balance");
  var riskInput = document.getElementById("wb-risk");
  var segBtns = Array.prototype.slice.call(document.querySelectorAll(".seg-btn"));

  function fmtPrice(val, decimals) {
    return Number(val).toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function updateWorkbench() {
    if (!form) return;
    var sym = symInput.value || "EUR/USD";
    var market = MARKETS[sym] || MARKETS["EUR/USD"];
    var side = sideInput.value === "SELL" ? "SELL" : "BUY";
    var horizon = Math.max(1, Math.min(120, parseInt(horInput.value || "15", 10) || 15));
    var entry = Math.max(0.0001, parseFloat(entryInput.value || String(market.entry)) || market.entry);
    var atr = Math.max(0.05, parseFloat(atrInput.value || String(market.atr)) || market.atr);
    var balance = Math.max(100, parseFloat(balInput.value || "10000") || 10000);
    var rawRiskPct = Math.max(0.1, parseFloat(riskInput.value || "1.0") || 1.0);
    var cappedRiskFrac = Math.min(rawRiskPct / 100.0, 0.02); // 2% ceiling matching fusion/risk.py

    var spread = market.spread;
    // Expected gross move over horizon bars under random-walk scaling (matches fusion/costs.py)
    var expectedMove = atr * Math.sqrt(horizon) * HALF_NORMAL_SCALE;
    // Break-even win rate: p*E - (1-p)*E - spread = 0 => p = 0.5 + spread / (2*E)
    var breakEven = expectedMove > 0 ? 0.5 + (spread / (2.0 * expectedMove)) : Infinity;

    // Minimum viable horizon for 3.0x reward-to-cost: atr * sqrt(h) * 0.7979 >= 3 * spread
    var neededMove = 3.0 * spread;
    var minHoldRaw = Math.ceil(Math.pow(neededMove / (atr * HALF_NORMAL_SCALE), 2));
    var minHoldText = minHoldRaw <= 120 ? (Math.max(1, minHoldRaw) + " min") : "> 120 min (too quiet)";

    // Hard ATR stop (1.5x ATR) & target (2.25x ATR), matching fusion/risk.py
    var stopPips = Math.max(spread * 1.5, atr * 1.5);
    var targetPips = Math.max(stopPips, atr * 2.25);
    var stopDelta = stopPips * market.pip;
    var targetDelta = targetPips * market.pip;

    var stopPrice = side === "BUY" ? (entry - stopDelta) : (entry + stopDelta);
    var targetPrice = side === "BUY" ? (entry + targetDelta) : (entry - targetDelta);

    var effRiskPips = stopPips + spread;
    var effRewardPips = Math.max(0, targetPips - spread);
    var netRR = effRiskPips > 0 ? (effRewardPips / effRiskPips) : 0;

    var riskUsd = balance * cappedRiskFrac;
    var lossPerUnitUsd = effRiskPips * market.pip;
    var units = lossPerUnitUsd > 0 ? (riskUsd / lossPerUnitUsd) : 0;

    document.getElementById("out-spread").textContent = spread.toFixed(2) + " pips";
    document.getElementById("out-be").textContent = breakEven <= 1.0 ? ((breakEven * 100).toFixed(1) + "%") : "Unpayable (>100%)";
    document.getElementById("out-minhold").textContent = minHoldText;
    document.getElementById("out-stop").textContent = fmtPrice(stopPrice, market.precision);
    document.getElementById("out-target").textContent = fmtPrice(targetPrice, market.precision);
    document.getElementById("out-rr").textContent = netRR.toFixed(2) + " : 1";
    document.getElementById("out-riskusd").textContent = "$" + fmtPrice(riskUsd, 2) + " (" + (cappedRiskFrac * 100).toFixed(2) + "%)";

    var unitDecimals = market.baseUnit === "EUR" ? 0 : 4;
    document.getElementById("out-units").textContent = fmtPrice(units, unitDecimals) + " " + market.baseUnit;

    var banner = document.getElementById("wb-verdict");
    if (expectedMove >= neededMove) {
      banner.textContent = sym + " at " + horizon + "m clears the 3.0x cost gate (projected " + expectedMove.toFixed(2) + " pips vs " + neededMove.toFixed(2) + " pip threshold).";
    } else if (minHoldRaw <= 120) {
      banner.textContent = sym + " at " + horizon + "m is below the 3.0x cost floor — adaptive horizon stretches hold to " + minHoldRaw + "m.";
    } else {
      banner.textContent = sym + " at " + atr.toFixed(2) + " pips ATR is too quiet for any horizon up to 120m — engine abstains.";
    }
  }

  segBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var sym = btn.getAttribute("data-symbol");
      var m = MARKETS[sym];
      if (!m) return;
      segBtns.forEach(function (b) { b.classList.remove("is-active"); });
      btn.classList.add("is-active");
      symInput.value = sym;
      entryInput.value = m.entry;
      atrInput.value = m.atr;
      horInput.value = m.horizon;
      updateWorkbench();
    });
  });

  if (form) {
    form.addEventListener("input", updateWorkbench);
    form.addEventListener("change", updateWorkbench);
    updateWorkbench();
  }

  /* 3. Optional live telemetry via relative /api/status ------------------ */
  fetch("api/status", { headers: { "Accept": "application/json" } })
    .then(function (res) { return res.ok ? res.json() : null; })
    .then(function (data) {
      if (!data || !data.engine) return;
      var badge = document.getElementById("live-badge");
      if (badge) {
        badge.textContent = "Engine Online · " + data.markets.length + " Markets Ready";
      }
      if (data.tests_passing) {
        var st = document.getElementById("stat-tests");
        if (st) {
          st.innerHTML = String(data.tests_passing) + ' <span class="unit">Passing</span>';
        }
      }
    })
    .catch(function () {
      /* Static hosting (GitHub Pages): keep verified baseline text cleanly. */
    });
})();
