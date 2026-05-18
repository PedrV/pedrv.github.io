/* Bio collapsible toggle */
function toggleBio() {
    const btn      = document.querySelector(".bio-toggle");
    const content  = document.getElementById("bio-collapsible");
    if (!btn || !content) return;

    const isOpen = content.classList.contains("open");

    if (isOpen) {
        content.classList.remove("open");
        btn.classList.remove("open");
        /* Wait for transition then hide from layout */
        setTimeout(function() {
            content.style.display = "none";
        }, 350);
    } else {
        content.style.display = "block";
        /* Tiny delay lets display:block apply before transition fires */
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                content.classList.add("open");
                btn.classList.add("open");
            });
        });
    }
}