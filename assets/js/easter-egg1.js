/* ============================================================
EASTER EGG — Click portrait X times to trigger Jean-Paul
Fires a Google Analytics event to track discoveries
============================================================ */
(function () {
    const CLICKS_REQUIRED = 7;
    const SWAP_DURATION = 4000;       // ms the easter egg image shows
    const GLITCH_DURATION = 600;      // ms the glitch animation runs
    const JEAN_PAUL_SRC = "assets/images/etc/Jean-Paul.png";
    const PORTRAIT_IDS = ["portrait-mini", "portrait-mini-mobile", "portrait-main", "portrait-main-mobile"];

    /* Inject glitch keyframes once into the page */
    const glitchStyle = document.createElement("style");
    glitchStyle.textContent = `
        @keyframes glitch {
            0%   { transform: translate(0); filter: none; }
            15%  { transform: translate(-4px, 2px); filter: hue-rotate(90deg) saturate(3); }
            30%  { transform: translate(4px, -2px); filter: hue-rotate(180deg) brightness(1.5); }
            45%  { transform: translate(-3px, -3px); filter: hue-rotate(270deg) saturate(5); }
            60%  { transform: translate(3px, 3px); filter: hue-rotate(0deg) brightness(0.7); }
            75%  { transform: translate(-2px, 2px); filter: saturate(0); }
            100% { transform: translate(0); filter: none; }
        }
        .jp-glitch {
            animation: glitch ${GLITCH_DURATION}ms steps(1) forwards;
        }
    `;
    document.head.appendChild(glitchStyle);

    let clickCount = 0;
    let resetTimer = null;
    let active = false;

    function getPortraits() {
        /* Collect whichever portrait imgs are currently in the DOM */
        return PORTRAIT_IDS.map(id => document.getElementById(id)).filter(Boolean);
    }

    function triggerEasterEgg() {
        if (active) return;
        active = true;
        clickCount = 0;

        const portraits = getPortraits();
        const origSrcs = portraits.map(img => img.src);
        const origStyles = portraits.map(img => img.getAttribute("style") || "");

        /* Fire GA event */
        if (typeof gtag === "function") {
            gtag("event", "easter_egg_discovered", {
                egg_id: "jean_paul"
            });
        }

        /* Glitch phase */
        portraits.forEach(img => img.classList.add("jp-glitch"));

        setTimeout(function () {
            /* Swap to Jean-Paul */
            portraits.forEach(function (img) {
                img.classList.remove("jp-glitch");
                img.src = JEAN_PAUL_SRC;
                /* Remove border-radius so Jean-Paul renders cleanly */
                img.style.borderRadius = "0.375rem";
            });

            /* Restore after SWAP_DURATION */
            setTimeout(function () {
                portraits.forEach(function (img, i) {
                    img.classList.add("jp-glitch");
                });

                setTimeout(function () {
                    portraits.forEach(function (img, i) {
                        img.classList.remove("jp-glitch");
                        img.src = origSrcs[i];
                        img.setAttribute("style", origStyles[i]);
                    });
                    active = false;
                }, GLITCH_DURATION);

            }, SWAP_DURATION);

        }, GLITCH_DURATION);
    }

    /* Attach click listeners after DOM is ready */
    document.addEventListener("DOMContentLoaded", function () {
        getPortraits().forEach(function (img) {
            img.style.cursor = "pointer";
            img.addEventListener("click", function () {
                if (active) return;

                clickCount++;

                /* Reset counter if user pauses for more than 2 seconds between clicks */
                clearTimeout(resetTimer);
                resetTimer = setTimeout(function () { clickCount = 0; }, 2000);

                if (clickCount >= CLICKS_REQUIRED) {
                    clearTimeout(resetTimer);
                    triggerEasterEgg();
                }
            });
        });
    });
})();